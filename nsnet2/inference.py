from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import json
import numpy as np
import torch
import soundfile as sf
from rich.progress import track

from common.env import AttrDict
from common.dataset import mag_pha_stft, mag_pha_istft, load_voicebank_demand
from nsnet2.model import NSNet2

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=True)
    print("Complete.")
    return checkpoint_dict


def enhance(model, noisy_wav):
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / (torch.sum(noisy_wav ** 2.0) + 1e-8)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, _ = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return (audio_g / norm_factor).squeeze().cpu().numpy()


def inference(a):
    model = NSNet2(h).to(device)
    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    os.makedirs(a.output_dir, exist_ok=True)

    with torch.no_grad():
        if a.input_noisy_wavs_dir:
            import librosa
            test_indexes = sorted(os.listdir(a.input_noisy_wavs_dir))
            for index in track(test_indexes):
                noisy_wav, _ = librosa.load(
                    os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate)
                audio_g = enhance(model, noisy_wav)
                sf.write(os.path.join(a.output_dir, index), audio_g, h.sampling_rate, 'PCM_16')
        else:
            hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
            test_split = hf['test']
            for item in track(test_split):
                noisy_wav = np.asarray(item['noisy']['array'], dtype=np.float32)
                audio_g = enhance(model, noisy_wav)
                sf.write(os.path.join(a.output_dir, item['id'] + '.wav'),
                         audio_g, h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default=None,
                        help='Directory of noisy wavs. If not set, uses HuggingFace test split.')
    parser.add_argument('--hf_cache_dir', default=None)
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
