# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 00:35:04 2021

@author: 82520
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:16:16 2021

@author: 82520
"""
import scipy.io.wavfile as wav

from parameters import *
from dataloader import read_f0, read_lo, read_waveform, smooth_scale_loudness
from synthesis import synthetize, reverb
from lossGPU import MSSLoss
from net2 import DDSPNet
#from dataloader import Dataset,DatasetVAL,DatasetTEST
from synthesis import synthetize
from timing import print_time, print_info
#from loss import compute_stft, spectral_loss
from loss import melspectral_loss,melspectrogram
from torch.utils.data import DataLoader
from torch import optim
from reverb import add_reverb
from parameters import *
from tensorboardX import SummaryWriter
from torch import nn
import scipy.io.wavfile as wav
import pdb
import torchaudio
import librosa
def create(checkpoint_path,fragmentpath,wavepath):
    Net = DDSPNet().float()
    Net = nn.DataParallel(Net,device_ids=[0,1])
    Net = Net.to(DEVICE)
    #model = build_model().to(DEVICE)
    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if GPU_ON:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    Net.load_state_dict(checkpoint) #["state_dict"]
    #print(pytorch_total_params = sum(p.numel() for p in Net.parameters()))
    
    
    
# =============================================================================
#     fragments, waveforms = torch.load(fragmentpath)
#     fragments['spe']=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(fragments['spe'].unsqueeze(0).unsqueeze(0))
#     fragments['spe']=fragments['spe'].squeeze(0)
#     fragments["f0"]=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(fragments["f0"].unsqueeze(0).unsqueeze(0))
#     fragments["f0"]=fragments["f0"].squeeze(0) 
# =============================================================================


    
    waveform, sr = torchaudio.load(wavepath)
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE,n_fft=N_FFT,f_min =F_MIN,f_max=F_MAX,
                                                   win_length=N_FFT,hop_length=HOP_LENGTH, n_mels=Binsize,pad=0)(waveform)
    f0=librosa.yin(waveform.numpy()[0],fmin=F0_MIN,fmax=F0_MAX,sr=AUDIO_SAMPLE_RATE,frame_length=N_FFT,win_length=None,
                         hop_length=HOP_LENGTH,trough_threshold=0.1,center=True,pad_mode='reflect')
    f0=torch.from_numpy(f0).unsqueeze(0)
    f0=f0.permute(1,0)
    mel_specgram=mel_specgram.permute(0,2,1)
    mel_specgram=mel_specgram.squeeze(0)   
    fragments={}
    fragments['spe']=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(mel_specgram.unsqueeze(0).unsqueeze(0))
    fragments['spe']=fragments['spe'].squeeze(0)
    fragments["f0"]=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(f0.unsqueeze(0).unsqueeze(0))
    fragments["f0"]=fragments["f0"].squeeze(0) 



    fragments['spe']=fragments['spe'].to(DEVICE)
    fragments["f0"] = fragments["f0"].to(DEVICE)   


    y_additive, y_noise = Net(fragments['spe'])



    f0s=fragments["f0"].squeeze(2)
    #f0s =f0s[:, :, 0]  
    a0s = y_additive[:, :, 0]   
    
    aa = y_additive[:, :, 1:NUMBER_HARMONICS + 1]  
    
    hh = y_noise[:, :, 0:NUMBER_NOISE_BANDS + 1]   


    if NOISE_ON:
        additive, bruit = synthetize(a0s, f0s, aa, hh, FRAME_LENGTH,
                                     AUDIO_SAMPLE_RATE, DEVICE)
        sons = additive + bruit
    else:
        additive, bruit = synthetize(a0s, f0s, aa, hh, FRAME_LENGTH,
                                     AUDIO_SAMPLE_RATE, DEVICE)
        sons = additive


    #wav.write('./wavsPPT/DDSPEECHour_46epoch_new.wav',AUDIO_SAMPLE_RATE, sons.detach().cpu().numpy()[0])
    #wav.write('./wavsPPT/DDSPEECHour_2epoch__new.wav',AUDIO_SAMPLE_RATE, sons.detach().cpu().numpy()[0])

    #wav.write('./wavsPPT/wavsglow/LJ001-0096our1.wav',AUDIO_SAMPLE_RATE, sons.detach().cpu().numpy()[0])

    #wav.write('./evaluation/otherpaper/LJ001-0153our_366epoch_new.wav',AUDIO_SAMPLE_RATE, sons.detach().cpu().numpy()[0])
    wav.write('./evaluation/otherpaper/LJ001-0096our_366epoch.wav',AUDIO_SAMPLE_RATE, sons.detach().cpu().numpy()[0])
if __name__ == "__main__":
    checkpoint_path=os.path.join(PATH_SAVED_MODELS, MODEL_CHECKPOINT + "366.pth")
    #wavepath='./evaluation/otherpaper/LJ001-0153real.wav'
    wavepath='./evaluation/otherpaper/LJ001-0096real.wav'
    
    
    
    #wavepath='./wavsPPT/wavsglow/LJ001-0153original.wav'
    #wavepath='./wavsPPT/wavsglow/LJ001-0096original.wav'
    #wavepath='./wavsPPT/wavs/DDSPEECHoriginal.wav'
    
    
    

    fragmentpath=' ' #'./FragmentVAL/1.pth'
    create(checkpoint_path,fragmentpath,wavepath)
