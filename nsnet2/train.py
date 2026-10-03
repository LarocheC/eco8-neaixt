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
from common.dataset import (Dataset, data_generator, load_voicebank_demand, mag_pha_istft,
                            mag_pha_stft, seed_worker)
from common.determinism import apply_determinism, rng_state, seed_everything, set_rng_state
from nsnet2.model import NSNet2
from nsnet2.bfly_arch import build_generator
from common.metrics import pesq_score
from common.discriminator import MetricDiscriminator, batch_pesq
from nsnet2.layers import butterfly_ortho_penalty
from nsnet2.sparsity import SparsityController
from nsnet2.optim_groups import build_param_groups, describe
from common.utils import scan_checkpoint, load_checkpoint, save_checkpoint

try:
    from torch_structured.butterfly.butterfly import Butterfly as _Butterfly
except ImportError:
    _Butterfly = None

# torch_structured's Triton butterfly backward launches its kernels through
# torch.library's wrap_triton, and in eager mode every launch registers the
# kernel and its constant args in PyTorch's global kernel_side_table, which
# nothing ever clears: ~8 KiB of host RAM per butterfly backward, ~4 MiB per
# NSNet2 step with a butterfly GRU (500 per-timestep calls), i.e. ~36 GiB per
# 200-epoch run -- three such runs got OOM-killed. The table only serves
# torch.compile tracing; this trainer is eager-only, so it is cleared after
# every step (a no-op for models without user Triton kernels).
try:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table as _kernel_side_table
except ImportError:
    _kernel_side_table = None

# Determinism over autotuning: benchmark picks algorithms nondeterministically
# per input shape, which (with the fixed seeds below) is the main remaining
# source of run-to-run drift. The input shapes here are static, so autotuning
# buys little.
torch.backends.cudnn.benchmark = False


class LinearWarmup:
    """Linear LR warmup for one optimizer, layered over a per-epoch scheduler.

    The first ``warmup_steps`` optimizer steps (global step index ``s``) run at
    ``lr_epoch * (s + 1) / warmup_steps``, where ``lr_epoch`` is the group's lr
    recorded by ``epoch_start()`` (i.e. whatever the per-epoch scheduler set);
    afterwards they run at ``lr_epoch`` itself. ``epoch_end()`` restores the
    unscaled lr BEFORE ``scheduler.step()``, so a chainable scheduler such as
    ExponentialLR (which multiplies the current group lr) never sees a scaled
    value and its per-epoch trajectory is exactly the no-warmup one.
    ``warmup_steps <= 0`` makes every method a no-op.
    """

    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps or 0)
        self.base_lrs = None

    @property
    def enabled(self):
        return self.warmup_steps > 0

    def epoch_start(self):
        if self.enabled:
            self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]

    def scale(self, step):
        if not self.enabled or step >= self.warmup_steps:
            return 1.0
        return (step + 1) / self.warmup_steps

    def before_step(self, step):
        """Set the lr used by the optimizer step at global step ``step``."""
        if not self.enabled:
            return
        if step >= self.warmup_steps:
            self._restore()
            return
        k = self.scale(step)
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = lr * k

    def epoch_end(self):
        if self.enabled:
            self._restore()

    def _restore(self):
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = lr


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    # Opt-in reproducibility. All three default to false, so a config written
    # before they existed trains bit for bit as it did -- running and queued
    # studies are unaffected. See common/determinism.py for what each fixes.
    det_on = apply_determinism(h.get("deterministic", False))
    use_data_gen = h.get("data_generator", False)
    exact_resume = h.get("exact_resume", False)
    if det_on:
        seed_everything(h.seed)
    if rank == 0 and (det_on or use_data_gen or exact_resume):
        print('Reproducibility: deterministic={} data_generator={} exact_resume={}'
              .format(det_on or False, use_data_gen, exact_resume))

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank)) if torch.cuda.is_available() else torch.device('cpu')

    generator = build_generator(h).to(device)   # NSNet2 unless h.arch is set
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

    # Butterfly twiddles can take their own learning rate and weight decay.
    # Both default to a no-op, so a config written before this trains exactly
    # as it did. See nsnet2/optim_groups.py for why either is worth varying.
    tw_mult = h.get("twiddle_lr_mult", 1.0)
    tw_wd = h.get("twiddle_weight_decay", None)
    g_groups = build_param_groups(generator, h.learning_rate,
                                  twiddle_lr_mult=tw_mult,
                                  twiddle_weight_decay=tw_wd,
                                  weight_decay=h.get("weight_decay", None))
    optim_g = torch.optim.AdamW(g_groups, h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if rank == 0 and (tw_mult != 1.0 or tw_wd is not None):
        for line in describe(g_groups):
            print('  optim_g {}'.format(line))
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

    # Test hook (default off): shorten the epoch so the determinism acceptance
    # tests can cross several epoch boundaries in a handful of steps. At the
    # real batch 256 an epoch is only ~45 batches, so epoch-boundary handling is
    # on the normal resume path and has to be covered.
    if h.get("train_subset", 0):
        trainset.indices = trainset.indices[:int(h["train_subset"])]
        if rank == 0:
            print('train_subset: epoch truncated to {} utterances'.format(len(trainset.indices)))

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    # Without a generator the loader draws its worker base seed from the global
    # RNG at iterator creation -- i.e. AFTER model init, so two architectures
    # with the same seed consume different amounts of RNG and see different
    # crops. A seeded generator pins the base seed; seed_worker then pins the
    # per-worker python/numpy RNGs that Dataset.__getitem__ crops with.
    data_gen = data_generator(h.seed) if use_data_gen else None
    loader_kw = dict(worker_init_fn=seed_worker, generator=data_gen) if use_data_gen else {}

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              **loader_kw)
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

    # Exact resume: how far into the epoch we got, plus every RNG stream as of
    # that batch. Checkpoints written before this existed carry no 'next_batch',
    # and fall back to the old behaviour of restarting the epoch from batch 0.
    resume_epoch = resume_skip = None
    pending_rng = epoch_entry_rng = None
    if exact_resume and state_dict_do is not None:
        if 'next_batch' in state_dict_do:
            resume_epoch = state_dict_do['epoch']
            resume_skip = state_dict_do['next_batch']
            pending_rng = state_dict_do.get('rng')
            epoch_entry_rng = state_dict_do.get('epoch_start_rng')
            if rank == 0:
                print('Exact resume: epoch {}, skipping {} batches already trained on'
                      .format(resume_epoch, resume_skip))
        elif rank == 0:
            print('Exact resume requested but this checkpoint predates it; '
                  'restarting epoch {} from its first batch'.format(last_epoch))
    quant_first_cycle_done = False    # Phase 6 (TRN-03)

    # Generator-only linear LR warmup (h.warmup_steps, default 0 = off).
    warmup_g = LinearWarmup(optim_g, h.get("warmup_steps", 0))
    if rank == 0 and warmup_g.enabled:
        print('Generator LR warmup: {} steps'.format(warmup_g.warmup_steps))

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        # Rewind before the iterator exists: the worker base seed is drawn when
        # the DataLoader iterator is created, once per epoch, so restoring the
        # epoch-start snapshot here is what makes the workers replay this
        # epoch's crops in the same order.
        skip = 0
        if resume_epoch is not None and epoch == resume_epoch:
            skip = resume_skip
            set_rng_state(epoch_entry_rng, data_gen)
        epoch_start_rng = rng_state(data_gen)

        warmup_g.epoch_start()

        # Drive the iterator by hand. The batches already trained on must be
        # replayed -- that is what walks the loader's workers forward to where
        # they were -- and the checkpointed RNG must go back BEFORE the next
        # batch is drawn. With num_workers=0 there are no workers and the crop
        # happens inside next(), so restoring from inside the loop body (i.e.
        # after the draw) would rewind the stream by one batch.
        train_iter = iter(train_loader)
        for _ in range(skip):
            try:
                next(train_iter)
            except StopIteration:      # the cut fell on the epoch's last batch
                break
        if pending_rng is not None:
            set_rng_state(pending_rng, data_gen)
            pending_rng = None

        for i, batch in enumerate(train_iter, start=skip):
            if a.max_steps and steps >= a.max_steps:
                break

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
            warmup_g.before_step(steps)
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
            if _kernel_side_table is not None:
                _kernel_side_table.reset_table()
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
                                     'best_pesq': best_pesq,
                                     # Written unconditionally (a few KB) so a run
                                     # started without exact_resume can still be
                                     # resumed exactly later.
                                     'next_batch': i + 1,
                                     'rng': rng_state(data_gen),
                                     'epoch_start_rng': epoch_start_rng})

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

                        # Composed-spectrum tracking (h.log_spectra, default off).
                        # The rank collapse in these models is present AT INIT and
                        # training raises rank from there, so the spectrum has to be
                        # tracked rather than inspected post hoc. Logged here because
                        # it is also exactly what the later quantisation phase needs,
                        # and collecting it now costs one validation's worth of time
                        # instead of re-running every arm.
                        if h.get("log_spectra", False):
                            from nsnet2.diagnostics import layer_spectra
                            gen_inner = generator.module if h.num_gpus > 1 else generator
                            for lname, sp in layer_spectra(gen_inner).items():
                                if "error" in sp:
                                    continue
                                sw.add_scalar("Cond/{}".format(lname), sp["cond"], steps)
                                sw.add_scalar("Rank99/{}".format(lname), sp["rank99"], steps)
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

        if a.max_steps and steps >= a.max_steps:
            # Stop inside the epoch, before the schedulers step: a partial epoch
            # must not advance the per-epoch LR schedule.
            if rank == 0:
                print('Reached max_steps={}; stopping.'.format(a.max_steps))
            break

        warmup_g.epoch_end()     # unscaled lr back before the scheduler reads it
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
    parser.add_argument('--max_steps', default=0, type=int,
                        help='Stop once this many optimizer steps have run '
                             '(0 = no limit). Used by the determinism '
                             'acceptance tests to land on an exact step.')

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
