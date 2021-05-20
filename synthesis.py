import torch.nn.functional as func
import numpy as np
import scipy.io.wavfile as wav

from noise import synthetize_noise
from dataloader import int_2_float
from parameters import *


def synthetize(a0s, f0s, aa, hh, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]
    nb_bounds = f0s.size()[1]
    frame_length=30#257 #1024/4
    signal_length = (nb_bounds - 1) * frame_length
    #signal_length=f0s.size()[1]*256
    nb_harms = aa.size()[-1]


    harm_ranks = torch.arange(nb_harms, device=device) + 1
    ff = f0s.unsqueeze(-1) * harm_ranks.unsqueeze(0).unsqueeze(0)
    max_f = sample_rate / 2.1
    aa[ff >= max_f] = 0.0

    f0s = f0s.unsqueeze(1)
    f0s = func.interpolate(f0s, size=signal_length, mode="linear",
                           align_corners=True)
    f0s = f0s.squeeze(1)
    phases = 2 * np.pi * f0s / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)
    phases_acc = phases_acc.unsqueeze(-1) * harm_ranks
    aa_sum = torch.sum(aa, dim=2)
    aa_sum[aa_sum == 0.] = 1.
    aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa_norm * a0s.unsqueeze(-1)
    aa = smoothing_amplitudes(aa, signal_length, HAMMING_WINDOW_LENGTH, device)
    additive = aa * torch.sin(phases_acc)
    additive = torch.sum(additive, dim=2)
    noise = synthetize_noise(hh, device)
    torch.cuda.empty_cache()

    return additive, noise


def smoothing_amplitudes(aa, signal_length, window_length, device):
    aa = func.interpolate(aa.transpose(1, 2), size=signal_length,
                          mode='linear', align_corners=True)
    aa = aa.transpose(1, 2)

    if HAMMING_SMOOTHING:
        aa_downsampled = aa[:, ::window_length, :]
        aa_interpolated = interpolate_hamming(aa_downsampled, signal_length,
                                              window_length, device)

        return aa_interpolated
    else:
        return aa


def prevent_aliasing(ff, aa, f_max, f_min):
    temp = (f_max - ff[ff >= f_min]) / (f_max - f_min)
    aa[ff >= f_min] *= temp**4
    aa[ff >= f_max] = 0

    return aa


def interpolate_hamming(tensor, signal_length, frame_length, device):
    y = torch.zeros((tensor.shape[0], tensor.shape[1] * frame_length,
                     tensor.shape[2]), device=device)
    y[:, ::frame_length, :] = tensor

    y = torch.transpose(y, 1, 2)
    y_padded = func.pad(y, [frame_length+1, frame_length])
    hamming_length = 2*frame_length+1
    w = torch.hamming_window(hamming_length, device=device, periodic=False)
    w = w.expand(y.shape[1], 1, hamming_length)

    interpolation = torch.conv1d(y_padded, w, groups=y_padded.shape[1])
    interpolation = torch.transpose(interpolation[:, :, 0:signal_length], 1, 2)

    return interpolation
