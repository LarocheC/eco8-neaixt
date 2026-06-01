import random
import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset


HF_DATASET_NAME = "JacobLinCool/VoiceBank-DEMAND-16k"


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-10, stft_spec[:, :, :, 0] + 1e-5)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)
    return wav


def load_voicebank_demand(cache_dir=None):
    """Load JacobLinCool/VoiceBank-DEMAND-16k (train + test splits)."""
    return load_dataset(HF_DATASET_NAME, cache_dir=cache_dir)


def seed_worker(worker_id):
    """Deterministic per-worker seeding for DataLoader workers.

    ``Dataset.__getitem__`` crops a random window with the global ``random``
    module; without this the crop (and any numpy RNG) is nondeterministic
    across runs/resumes. Seeds ``random`` and numpy from torch's per-worker
    base seed. Pair with ``data_generator(seed)`` as the loader's
    ``generator=`` so the base seed itself is fixed regardless of how much RNG
    the main process consumed before iterating.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_generator(seed):
    """A torch.Generator seeded for reproducible DataLoader shuffling/crops."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class Dataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace VoiceBank-DEMAND-16k split.

    Each row exposes paired ``clean`` and ``noisy`` audio decoded to numpy
    arrays at 16 kHz. The clean/noisy lengths always match by construction,
    so we crop a random ``segment_size`` window during training and return
    the full utterance during validation.
    """

    def __init__(self, hf_split, segment_size, sampling_rate, split=True,
                 shuffle=True, seed=1234):
        self.hf_split = hf_split
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split

        self.indices = list(range(len(hf_split)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.hf_split[self.indices[index]]
        clean_audio = np.asarray(item["clean"]["array"], dtype=np.float32)
        noisy_audio = np.asarray(item["noisy"]["array"], dtype=np.float32)

        length = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:length]
        noisy_audio = noisy_audio[:length]

        clean_audio = torch.from_numpy(clean_audio)
        noisy_audio = torch.from_numpy(noisy_audio)

        norm_factor = torch.sqrt(len(noisy_audio) / (torch.sum(noisy_audio ** 2.0) + 1e-8))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                pad = self.segment_size - clean_audio.size(1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad), 'constant')

        return clean_audio.squeeze(0), noisy_audio.squeeze(0)
