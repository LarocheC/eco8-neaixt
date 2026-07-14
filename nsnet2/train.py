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
from common.discriminator import MetricDiscriminator, pesq_target_from_config
from common.losses import (
    discriminator_loss, generator_loss, generator_terms, loss_weights,
)
from nsnet2.layers import butterfly_ortho_penalty
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
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

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

    # MP-SENet's generator weights; a config's `loss` block can override any of
    # them. `phase` is unused here — NSNet2 has no phase decoder.
    weights = loss_weights(h)
    pesq_fn = pesq_target_from_config(h)

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

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            batch_pesq_score = pesq_fn(clean_audio.detach(), audio_g.detach())

            # Discriminator (MetricGAN: predict normalised PESQ; clean-vs-clean = 1)
            optim_d.zero_grad()
            if batch_pesq_score is None:
                print('pesq is None!')
            loss_disc_all = discriminator_loss(
                discriminator, clean_mag, mag_g_hat, batch_pesq_score, device)
            loss_disc_all.backward()
            optim_d.step()

            # Generator — the MP-SENet objective. NSNet2 reuses the noisy phase
            # (no phase decoder), so pha_g=None drops the anti-wrapping phase term.
            optim_g.zero_grad()
            metric_g = discriminator(clean_mag, mag_g_hat)
            terms = generator_terms(
                mag_g=mag_g, com_g=com_g, mag_r=clean_mag, com_r=clean_com,
                com_g_hat=com_g_hat, audio_g=audio_g, audio_r=clean_audio,
                metric_g=metric_g,
            )
            loss_gen_all = generator_loss(terms, weights)

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
            optim_g.step()

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = float(terms['metric'])
                        mag_error = float(terms['magnitude'])
                        com_error = float(terms['complex'])
                        time_error = float(terms['time'])
                        stft_error = float(terms['consistency'])
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
