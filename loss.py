import torch
from torch.nn import functional as func
import torchaudio
from torchaudio.transforms import MelSpectrogram
import librosa
from parameters import *

    

def melspectrogramTORCH(inputs,n_fft):
    melspec = MelSpectrogram(AUDIO_SAMPLE_RATE,n_fft=n_fft,f_min =0,f_max=8000, 
                                  n_mels=80,win_length=n_fft,hop_length=int(n_fft/4) ,
                                  pad=0)(inputs)    
    return melspec

def melspectral_loss(synthesis, truth, fft_sizes):
    losses = torch.zeros(len(fft_sizes))

    for i, fft_size in enumerate(fft_sizes):
        mel_synthesis = MelSpectrogram(AUDIO_SAMPLE_RATE,n_fft=fft_size,f_min =0,f_max=8000, 
                                      n_mels=80,win_length=fft_size,hop_length=int(fft_size/4) ,
                                      pad=0)(synthesis)
             
        mel_truth = MelSpectrogram(AUDIO_SAMPLE_RATE,n_fft=fft_size,f_min =0,f_max=8000, 
                                      n_mels=80,win_length=fft_size,hop_length=int(fft_size/4) ,
                                      pad=0)(truth)
         
        eps = torch.finfo(mel_synthesis.dtype).eps

        mel_synthesis_log = torch.log(mel_synthesis + eps)
        mel_truth_log = torch.log(mel_truth + eps)


# =============================================================================
#         mel_synthesis=melspectrogram(synthesis,fft_size)
#         mel_truth=melspectrogram(truth,fft_size)
#         mel_synthesis_log=logmelspectrogram(synthesis,fft_size)
#         mel_truth_log=logmelspectrogram(truth,fft_size)
# =============================================================================

        loss_lin = func.l1_loss(mel_synthesis, mel_truth, reduction="mean")
        loss_log = func.l1_loss(mel_synthesis_log, mel_truth_log, reduction="mean")
        losses[i] = loss_lin + loss_log

    loss = torch.mean(losses)
    return loss










# =============================================================================
# def spectrogram(inputs,n_fft=2048, alpha=1.0, overlap=0.75, eps=1e-7):
#     hop_length = int(n_fft * (1 - overlap))  # 25% of the length
#     spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(inputs))
#     rerurn spec
# =============================================================================



# =============================================================================
# def spectral_loss(stft_all, stft_truth_all, fft_sizes):
#     losses = torch.zeros(len(fft_sizes))
# 
#     for i, fft_size in enumerate(fft_sizes):
#         stft = stft_all[fft_size]
#         stft_truth = stft_truth_all[fft_size]
# 
#         eps = torch.finfo(stft.dtype).eps
# 
#         stft_log = torch.log(stft + eps)
#         stft_truth_log = torch.log(stft_truth + eps)
# 
#         loss_lin = func.l1_loss(stft, stft_truth, reduction="mean")
#         loss_log = func.l1_loss(stft_log, stft_truth_log, reduction="mean")
# 
#         losses[i] = loss_lin + loss_log
# 
#     loss = torch.mean(losses)
#     return loss
# 
# 
# def compute_stft(waveform, fft_sizes):
#     stft_all = {}
# 
#     for fft_size in fft_sizes:
#         stft = torch.stft(waveform, fft_size, hop_length=fft_size // 4,
#                           center=True, pad_mode='reflect',
#                           normalized=False, onesided=True)
#         stft = torch.sum(stft**2, dim=-1)
#         stft_all[fft_size] = stft
# 
#     return stft_all
# 
# =============================================================================




import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
#from nnmnkwii import preprocessing as P


def low_cut_filter(x, fs, cutoff=70):
    """APPLY LOW CUT FILTER.

    https://github.com/kan-bayashi/PytorchWaveNetVocoder

    Args:
        x (ndarray): Waveform sequence.
        fs (int): Sampling frequency.
        cutoff (float): Cutoff frequency of low cut filter.
    Return:
        ndarray: Low cut filtered waveform sequence.
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    from scipy.signal import firwin, lfilter

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


# =============================================================================
# def load_wav(path):
#     sr, x = wavfile.read(path)
#     signed_int16_max = 2**15
#     if x.dtype == np.int16:
#         x = x.astype(np.float32) / signed_int16_max
#     if sr != hparams.sample_rate:
#         x = librosa.resample(x, sr, hparams.sample_rate)
#     x = np.clip(x, -1.0, 1.0)
#     return x
# 
# 
# def save_wav(wav, path):
#     wav *= 32767 / max(0.01, np.max(np.abs(wav)))
#     wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))
# 
# 
# def trim(quantized):
#     start, end = start_and_end_indices(quantized, hparams.silence_threshold)
#     return quantized[start:end]
# 
# 
# def preemphasis(x, coef=0.85):
#     return P.preemphasis(x, coef)
# 
# 
# def inv_preemphasis(x, coef=0.85):
#     return P.inv_preemphasis(x, coef)
# =============================================================================


def adjust_time_resolution(quantized, mel):
    """Adjust time resolution by repeating features

    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)

    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)

    return quantized[start:end], mel[start:end, :]


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def logmelspectrogram(y, fft_size,pad_mode="reflect"):
    """Same log-melspectrogram computation as espnet
    https://github.com/espnet/espnet
    from espnet.transform.spectrogram import logmelspectrogram
    """
    D = _stft(y,fft_size, pad_mode=pad_mode)
    S = _linear_to_mel(np.abs(D))
    S = np.log10(np.maximum(S, 1e-10))
    return S

def melspectrogram(y, fft_size,pad_mode="reflect"):
    """Same log-melspectrogram computation as espnet
    https://github.com/espnet/espnet
    from espnet.transform.spectrogram import logmelspectrogram
    """
    D = _stft(y,fft_size, pad_mode=pad_mode)
    S = _linear_to_mel(np.abs(D))
    return S



def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def get_win_length():
    win_length = hparams.win_length
    if win_length < 0:
        assert hparams.win_length_ms > 0
        win_length = int(hparams.win_length_ms / 1000 * hparams.sample_rate)
    return win_length


def _stft(y, fft_size,pad_mode="constant"):
    # use constant padding (defaults to zeros) instead of reflection padding
    return librosa.stft(y=y, n_fft=fft_size, hop_length=int(fft_size/4),
                        win_length=fft_size, window='hann',pad_mode=pad_mode)

        
def pad_lr(x, fsize, fshift):
    return (0, fsize)

# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
# =============================================================================
#     if hparams.fmax is not None:
#         assert hparams.fmax <= hparams.sample_rate // 2
# =============================================================================
    return librosa.filters.mel(AUDIO_SAMPLE_RATE, hparams.fft_size,
                               fmin=F_MIN, fmax=F_MAX,
                               n_mels=Binsize)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


