import torch
import torchaudio

from scipy.io.wavfile import read
from torch.utils.data import Dataset as ParentDataset
from parameters import *
import librosa
#import torchcrepe
class dataset(ParentDataset):
    def __init__(self):
        if not os.path.isdir(FRAGMENT_PATH):
            os.makedirs(FRAGMENT_PATH)
        self.compute_frag()

    def __len__(self):
        return self.nb_frags
    
    def __getitem__(self,idx):
        frag_path=os.path.join(FRAGMENT_PATH,str(idx)+".pth")
        frag=torch.load(frag_path)
        return frag
    
    def compute_frag(self):
        audio_files=os.listdir(AUDIO_PATH)
        n_files=len(audio_files)
        idx_i=0
        for file in audio_files:
            print(file)
            if file[0]=='.':
                continue
            file_path=os.path.join(AUDIO_PATH,file)
            waveform, sr = torchaudio.load(file_path)
            #waveform=torchaudio.transforms.Resample(sr,AUDIO_SAMPLE_RATE)(waveform)
            mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE,n_fft=N_FFT,f_min =F_MIN,f_max=F_MAX,
                                                           win_length=N_FFT,hop_length=HOP_LENGTH, n_mels=Binsize,pad=0)(waveform)
        
            f0=librosa.yin(waveform.numpy()[0],fmin=F0_MIN,fmax=F0_MAX,sr=AUDIO_SAMPLE_RATE,frame_length=N_FFT,win_length=None,
                                 hop_length=HOP_LENGTH,trough_threshold=0.1,center=True,pad_mode='reflect')
            
            f0=torch.from_numpy(f0).unsqueeze(0)
            nb_frag=waveform.shape[1]//(AUDIO_SAMPLE_RATE*FRAGMENT_DURATION)
            for idx in range(int(nb_frag)):
                inputs,waveforms=self.separate_frag(mel_specgram,f0,idx,nb_frag,waveform)
                fragment_path=os.path.join(FRAGMENT_PATH,str(idx_i)+'.pth')
                torch.save((inputs,waveforms),fragment_path)
                idx_i+=1




            waveform=waveform[:,AUDIO_SAMPLE_RATE:]         
            mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE,n_fft=N_FFT,f_min =F_MIN,f_max=F_MAX,
                                                           win_length=N_FFT,hop_length=HOP_LENGTH, n_mels=Binsize,pad=0)(waveform)
            f0=librosa.yin(waveform.numpy()[0],fmin=F0_MIN,fmax=F0_MAX,sr=AUDIO_SAMPLE_RATE,frame_length=N_FFT,win_length=None,
                                 hop_length=HOP_LENGTH,trough_threshold=0.1,center=True,pad_mode='reflect')
            f0=torch.from_numpy(f0).unsqueeze(0)
            nb_frag=waveform.shape[1]//(AUDIO_SAMPLE_RATE*FRAGMENT_DURATION)
            for idx in range(int(nb_frag)):
                inputs,waveforms=self.separate_frag(mel_specgram,f0,idx,nb_frag,waveform)
                fragment_path=os.path.join(FRAGMENT_PATH,str(idx_i)+'.pth')
                torch.save((inputs,waveforms),fragment_path)
                idx_i+=1  

            



        self.nb_frags=idx_i
            

    def separate_frag(self,mel,f0,idx,nb_frag,waveform):
        
        mel_start=idx*FRAGMENT_MEL_LENGTH
        mel_end=(idx+1)*FRAGMENT_MEL_LENGTH
        sub_mel=mel[:,:,mel_start:mel_end]


        f0_start=idx*FRAGMENT_F0_LENGTH
        f0_end=(idx+1)*FRAGMENT_F0_LENGTH
        sub_f0=f0[:,f0_start:f0_end]


        waveform_stride = FRAGMENT_DURATION * AUDIO_SAMPLE_RATE
        waveform_start_i = idx * waveform_stride
        waveform_end_i = (idx + 1) * waveform_stride
        #waveform_end_i -= FRAME_LENGTH  #????
        
        sub_f0=sub_f0.permute(1,0)
        sub_mel=sub_mel.permute(0,2,1)
        sub_mel=sub_mel.squeeze(0)
        inputs = {"f0": sub_f0, "spe": sub_mel}
        waveforms=waveform[0][waveform_start_i:waveform_end_i]

        return inputs,waveforms


if  __name__=="__main__":
    dataset=dataset()
    print('true')
            

