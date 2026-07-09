"""Train the end-to-end butterfly NSNet2 (waveform -> waveform, no STFT in the model).

Adapted from ``nsnet2/train.py``. The generator is ``model_e2e.NSNet2E2E`` — a
learned butterfly analysis/synthesis around the unchanged NSNet2 core. There is
NO STFT in the model or the exported graph; the STFT only appears in the *loss*:

    loss = w_time   * L1(clean, esti)                       (time domain)
         + w_mrstft * multi-resolution compressed-mag MSE   (spectral)
         + w_metric * MetricGAN PESQ-proxy                  (adversarial)
         + lambda   * butterfly orthogonality penalty       (conditioning)

Validation PESQ is computed directly on the waveform output (no STFT round-trip).
"""

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
from common.dataset import Dataset, mag_pha_stft, load_voicebank_demand
from nsnet2.model_e2e import NSNet2E2E
from common.metrics import pesq_score
from common.discriminator import MetricDiscriminator, batch_pesq
from common.utils import scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = False

# (n_fft, hop, win) resolutions for the multi-resolution STFT magnitude loss.
DEFAULT_MRSTFT = [[512, 256, 512], [1024, 256, 1024], [256, 128, 256]]


def mrstft_mag_loss(clean, esti, resolutions, compress):
    """Mean compressed-magnitude MSE across STFT resolutions. STFT is used here
    purely as a *loss*; it never enters the model or the ONNX graph."""
    total = 0.0
    for n_fft, hop, win in resolutions:
        cm, _, _ = mag_pha_stft(clean, n_fft, hop, win, compress)
        gm, _, _ = mag_pha_stft(esti, n_fft, hop, win, compress)
        total = total + F.mse_loss(cm, gm)
    return total / len(resolutions)


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank)) if torch.cuda.is_available() else torch.device('cpu')

    generator = NSNet2E2E(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    resolutions = h.get("mrstft_resolutions", DEFAULT_MRSTFT)
    loss_w = h.get("loss", {})
    w_time = loss_w.get("time", 1.0)
    w_mrstft = loss_w.get("mrstft", 1.0)
    w_metric = loss_w.get("metric", 0.05)
    ortho_lambda = h.get("butterfly_ortho_lambda", 0.0)
    # Fixed STFT (loss-side only) that feeds the MetricGAN discriminator's magnitudes.
    d_nfft = h.get("disc_n_fft", 512)
    d_hop = h.get("disc_hop_size", 256)
    d_win = h.get("disc_win_size", 512)
    compress = h.get("compress_factor", 0.3)

    def disc_mag(wav):
        m, _, _ = mag_pha_stft(wav, d_nfft, d_hop, d_win, compress)
        return m

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

    best_pesq = state_dict_do.get('best_pesq', 0) if state_dict_do is not None else 0

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

            audio_g = generator(noisy_audio)                    # (B, L) waveform

            clean_mag = disc_mag(clean_audio)
            mag_g_hat = disc_mag(audio_g)

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

            loss_time = F.l1_loss(clean_audio, audio_g)
            loss_mrstft = mrstft_mag_loss(clean_audio, audio_g, resolutions, compress)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = (loss_time * w_time
                            + loss_mrstft * w_mrstft
                            + loss_metric * w_metric)

            # Butterfly orthogonality penalty (analysis + synthesis twiddles).
            ortho_loss = None
            if ortho_lambda > 0:
                gen_inner = generator.module if h.num_gpus > 1 else generator
                ortho_loss = gen_inner.ortho_penalty()
                loss_gen_all = loss_gen_all + ortho_lambda * ortho_loss

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        time_error = loss_time.item()
                        mrstft_error = loss_mrstft.item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric loss: {:4.3f}, MR-STFT Loss : {:4.3f}, Time Loss : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, loss_disc_all, metric_error, mrstft_error, time_error, time.time() - start_b))

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
                    sw.add_scalar("Training/MR-STFT Loss", mrstft_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    if ortho_loss is not None:
                        sw.add_scalar("Training/Butterfly Ortho Penalty", ortho_loss.item(), steps)

                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_time_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, noisy_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            noisy_audio = noisy_audio.to(device, non_blocking=True)

                            audio_g = generator(noisy_audio)
                            audios_r += torch.split(clean_audio, 1, dim=0)
                            audios_g += torch.split(audio_g, 1, dim=0)
                            val_time_err_tot += F.l1_loss(clean_audio, audio_g).item()

                        val_time_err = val_time_err_tot / (j + 1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                              format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Time Loss", val_time_err, steps)

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
    parser.add_argument('--checkpoint_path', default='cp_nsnet2_butterfly_e2e')
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
