import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from common.env import AttrDict, build_env
from common.dataset import Dataset, mag_pha_stft, mag_pha_istft, load_voicebank_demand
from nsnet2.model import NSNet2
from common.metrics import pesq_score
from common.discriminator import MetricDiscriminator, batch_pesq
from nsnet2.layers import butterfly_ortho_penalty
from nsnet2.sparsity import SparsityController
from common.utils import scan_checkpoint, load_checkpoint, save_checkpoint

try:
    from torch_structured.butterfly.butterfly import Butterfly as _Butterfly
except ImportError:
    _Butterfly = None

# Determinism over autotuning: benchmark picks algorithms nondeterministically
# per input shape, which (with the fixed seeds below) is the main remaining
# source of run-to-run drift. The input shapes here are static, so autotuning
# buys little.
torch.backends.cudnn.benchmark = False


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank)) if torch.cuda.is_available() else torch.device('cpu')

    generator = NSNet2(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        print(generator)
        num_params = sum(p.numel() for p in generator.parameters())
        print('Total Parameters: {:.3f}M'.format(num_params / 1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    cp_g = cp_do = None
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
        # Warm-start from a dense checkpoint (the "dense -> prune -> masked
        # fine-tune" recipe). Weights only: steps/epoch/optimizer stay fresh.
        if a.init_from:
            generator.load_state_dict(load_checkpoint(a.init_from, device)['generator'])
            if rank == 0:
                print('Warm-started generator from {}'.format(a.init_from))
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # Fixed structured-sparsity masks (h.sparsity). Built here — after the
    # checkpoint load, before DDP — so magnitude selection sees trained weights
    # and the controller holds the real Parameter objects (DDP wraps the module
    # but keeps the same parameter tensors).
    sparsity = SparsityController.from_config(generator, h.get("sparsity", None))
    if sparsity is not None:
        sparsity.apply()
        if rank == 0:
            print(sparsity)
            for row in sparsity.report():
                print('  {name:<28} {shape} sparsity={sparsity:.3f} '
                      'tail={tail_elements}'.format(**row))

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Resume the LR schedule exactly. Reconstructing with last_epoch advances
    # the schedule one step past the saved position (and clobbers the lr that
    # optim load_state_dict just restored), so for checkpoints that carry the
    # scheduler state we restore it and push get_last_lr() back into the
    # optimizer (load_state_dict alone does not update the param-group lr).
    # Older checkpoints lack these keys and keep the last_epoch behaviour.
    if state_dict_do is not None and 'scheduler_g' in state_dict_do:
        scheduler_g.load_state_dict(state_dict_do['scheduler_g'])
        scheduler_d.load_state_dict(state_dict_do['scheduler_d'])
        for grp, lr in zip(optim_g.param_groups, scheduler_g.get_last_lr()):
            grp['lr'] = lr
        for grp, lr in zip(optim_d.param_groups, scheduler_d.get_last_lr()):
            grp['lr'] = lr

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)

    trainset = Dataset(hf['train'], h.segment_size, h.sampling_rate,
                       split=True, shuffle=False if h.num_gpus > 1 else True, seed=h.seed)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = Dataset(hf['test'], h.segment_size, h.sampling_rate,
                           split=False, shuffle=False, seed=h.seed)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()

    # Restore best_pesq on resume so a resumed run can't overwrite g_best with
    # an inferior model (the checkpoint persists it; default 0 for fresh runs).
    best_pesq = state_dict_do.get('best_pesq', 0) if state_dict_do is not None else 0
    quant_first_cycle_done = False    # Phase 6 (TRN-03)

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            # Discriminator
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                print('pesq is None!')
                loss_disc_g = 0

            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # L2 Complex Loss (mag-only model: equivalent to mag loss weighted by phase coherence)
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # L2 Consistency Loss
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = (loss_mag * 0.9
                            + loss_com * 0.1
                            + loss_stft * 0.1
                            + loss_metric * 0.05
                            + loss_time * 0.2)

            # Butterfly orthogonality penalty (gated by h.butterfly_ortho_lambda).
            # Pulls each 2x2 twiddle factor toward orthogonality so the cumulative
            # log_n-stage butterfly stays spectrally bounded — keeps activation
            # magnitudes int8-friendly across stages. No-op when lambda=0 or
            # there are no Butterfly modules in the model.
            ortho_lambda = h.get("butterfly_ortho_lambda", 0.0)
            ortho_loss = None
            if ortho_lambda > 0 and _Butterfly is not None:
                gen_inner = generator.module if h.num_gpus > 1 else generator
                bf_terms = [butterfly_ortho_penalty(m.twiddle)
                            for m in gen_inner.modules()
                            if isinstance(m, _Butterfly)]
                if bf_terms:
                    ortho_loss = torch.stack(bf_terms).mean()
                    loss_gen_all = loss_gen_all + ortho_lambda * ortho_loss

            loss_gen_all.backward()
            if sparsity is not None:
                sparsity.mask_grads()
            optim_g.step()
            if sparsity is not None:
                # Re-project onto the mask: AdamW's momentum/decay would
                # otherwise drift pruned weights off exactly zero.
                sparsity.apply()

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric loss: {:4.3f}, Magnitude Loss : {:4.3f}, Complex Loss : {:4.3f}, Time Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, loss_disc_all, metric_error, mag_error, com_error, time_error, stft_error, time.time() - start_b))

                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch,
                                     'scheduler_g': scheduler_g.state_dict(), 'scheduler_d': scheduler_d.state_dict(),
                                     'best_pesq': best_pesq})

                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)
                    if ortho_loss is not None:
                        sw.add_scalar("Training/Butterfly Ortho Penalty",
                                      ortho_loss.item(), steps)

                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, noisy_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            noisy_audio = noisy_audio.to(device, non_blocking=True)

                            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            audios_r += torch.split(clean_audio, 1, dim=0)
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()
                            val_stft_err_tot += F.mse_loss(com_g, com_g_hat).item()

                        val_mag_err = val_mag_err_tot / (j + 1)
                        val_com_err = val_com_err_tot / (j + 1)
                        val_stft_err = val_stft_err_tot / (j + 1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                              format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)
                        # Phase 6 hook (TRN-01..05; lazy-import gate per TRN-05).
                        if h.get("quant", {}).get("enabled", False):
                            from nsnet2.quant_hook import run_quant_eval
                            run_quant_eval(generator, h, hf, sw, steps,
                                           ckpt_dir=a.checkpoint_path,
                                           fp32_pesq=val_pesq_score,
                                           log_breakdown=(not quant_first_cycle_done),
                                           num_gpus=h.num_gpus)
                            quant_first_cycle_done = True

                    if epoch >= a.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(best_checkpoint_path,
                                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--hf_cache_dir', default=None,
                        help='Optional cache directory for the HuggingFace dataset.')
    parser.add_argument('--checkpoint_path', default='cp_nsnet2')
    parser.add_argument('--config', default='')
    parser.add_argument('--init_from', default='',
                        help='Optional g_* checkpoint to warm-start the generator '
                             'from when checkpoint_path is empty (dense -> masked '
                             'fine-tune).')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / max(h.num_gpus, 1))
        print('Batch size per GPU :', h.batch_size)

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
