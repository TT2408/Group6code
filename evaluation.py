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
from dataloader import Dataset,DatasetVAL
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
#from spectrogram import calspectrogram


    

def evaluation(VALDataloader,checkpoint_path):
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
    
    
    
    iterator = iter(VALDataloader)
    fragments, waveforms = iterator.next() 
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

# =============================================================================
#             if REVERB:
#                 sons = add_reverb(sons, impulse_response)
# =============================================================================

           
    waveforms = waveforms.to(DEVICE)

    lossNET = MSSLoss(FFT_SIZES)
    lossNET=nn.DataParallel(lossNET,device_ids=[0,1])
    lossNET=lossNET.to(DEVICE) 
    loss2=lossNET(sons.to(dtype=torch.float32),waveforms[:, 0:sons.shape[1]])
    torch.cuda.empty_cache()
    return loss2

if __name__ == "__main__":
    VALDataset = DatasetVAL()
    VALDataloader = DataLoader(VALDataset,batch_size=BATCH_SIZE_VAL,
                            shuffle=SHUFFLE_DATALOADER)
    checkpoint_path=os.path.join(PATH_SAVED_MODELS, MODEL_CHECKPOINT +str(2)+  ".pth")
    loss=evaluation(VALDataloader,checkpoint_path)
