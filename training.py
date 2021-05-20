import time
import librosa
from net2 import DDSPNet
from dataloader import Dataset,DatasetVAL
from synthesis import synthetize

from timing import print_time, print_info
#from loss import compute_stft, spectral_loss
from loss import melspectral_loss,melspectrogramTORCH
from lossGPU import MSSLoss
from torch.utils.data import DataLoader
from torch import optim
from reverb import add_reverb
from scipy.io.wavfile import read as wave_read
from parameters import *
from tensorboardX import SummaryWriter
#from spectrogram import calspectrogram
from Myevaluation import evaluation
from torch import nn

def train(net,lossNET, dataloader,valdataloader,number_epochs, debug_level): #
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    time_start = time.time()

# =============================================================================
#     if REVERB:
#         rate, impulse_response = wave_read('ir.wav')
#         impulse_response = \
#             impulse_response.astype(float)/max(abs(impulse_response))
#         impulse_response = torch.from_numpy(impulse_response)
#         impulse_response = impulse_response.type(torch.float).to(DEVICE)
# =============================================================================
    writer = SummaryWriter('Graph')
    for epoch in range(number_epochs):
        print_info("#### Epoch " + str(epoch+1) + "/" + str(number_epochs)
                   + " ####", debug_level, "TRAIN")
        time_epoch_start = time.time()
        nb_batchs = len(dataloader)
        epoch_loss = 0
        for i, data in enumerate(dataloader, 0):
            time_device_start = time.time()
            print_info("## Data " + str(i + 1) + "/" + str(nb_batchs)
                       + " ##", debug_level, "RUN")
            time_data_start = time.time()
            
            optimizer.zero_grad()

            fragments, waveforms = data
            fragments['spe']=fragments['spe'].to(DEVICE)
            fragments["f0"] = fragments["f0"].to(DEVICE)

            print_time("Time to device :", debug_level, "DEBUG",
                       time_device_start, 6)

            time_pre_net = time.time()



            fragments['spe']=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(fragments['spe'].unsqueeze(0))
            fragments['spe']=fragments['spe'].squeeze(0)
            fragments["f0"]=nn.UpsamplingBilinear2d(scale_factor=(upsamplefactor,1))(fragments["f0"].unsqueeze(0))
            fragments["f0"]=fragments["f0"].squeeze(0)            


            y_additive, y_noise = net(fragments['spe'])

            time_pre_synth = print_time("Time through net :", debug_level,
                                        "INFO", time_pre_net, 3)


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

            time_post_synth = print_time("Time to synthetize :", debug_level,
                                         "INFO", time_pre_synth, 3)

           
            waveforms = waveforms.to(DEVICE)

            #loss=melspectral_loss(sons.to(dtype=torch.float32),waveforms[:, 0:sons.shape[1]],FFT_SIZES)
            
            #with torch.no_grad():
            loss=lossNET(sons.to(dtype=torch.float32),waveforms[:, 0:sons.shape[1]])

# =============================================================================
#              """ STFT's """
#             squared_modules_synth = compute_stft(sons, FFT_SIZES)
#             squared_module_truth = compute_stft(waveforms[:, 0:sons.shape[1]],
#                                                 FFT_SIZES)
#             loss = spectral_loss(squared_modules_synth, squared_module_truth,
#                                  FFT_SIZES)
# =============================================================================

            time_post_stft = print_time("Time to perform stfts :", debug_level,
                                        "INFO", time_post_synth, 3)

            """ Loss & Backpropagation """
            loss.sum().backward()
            optimizer.step()

            epoch_loss += loss
            #print(epoch_loss)
                      
        writer.add_scalar('Trainset_loss', epoch_loss, epoch)
        if epoch%SavemodelPerEpoch==0 and epoch>0:
            PATH_TO_CHECKPOINT = os.path.join(PATH_SAVED_MODELS, MODEL_CHECKPOINT +str(epoch)+  ".pth")
            torch.save(net.state_dict(), PATH_TO_CHECKPOINT)

        if epoch%valperepoch==0 and epoch>0:
            checkpoint_path=os.path.join(PATH_SAVED_MODELS, MODEL_CHECKPOINT +str(epoch)+  ".pth")
            val_loss=evaluation(valdataloader,checkpoint_path)
            print('thisos')
            writer.add_scalar('VALset_loss', val_loss.mean(), epoch)
            
        scheduler.step()
        torch.cuda.empty_cache()
        #print(loss)
        #print(loss.shape)
        print_info("\n\n", debug_level, "RUN")
        print_time("Time of the epoch :", debug_level, "TRAIN",
                   time_epoch_start, 3)
        print_info("Epoch Loss : " +
                   str(round(epoch_loss.mean().item() / nb_batchs, 3)),
                   debug_level, "TRAIN")
        print_info("\n\n\n------------\n\n\n", debug_level, "RUN")


    ''' Save '''
    
   # torch.save(net.state_dict())

    #print_time("Training time :", debug_level, "TRAIN", time_start, 3)

    return


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    print(torch.cuda.device_count())
    print(DEVICE)
    ''' Debug settings '''
    PRINT_LEVEL = "TRAIN"  # Possible modes : DEBUG, INFO, RUN, TRAIN
    print_info("Starting training with debug level : " + PRINT_LEVEL,
               PRINT_LEVEL, "TRAIN")

    ''' Pytorch settings '''
    torch.set_default_tensor_type(torch.FloatTensor)
    print_info("Working device : " + str(DEVICE), PRINT_LEVEL, "INFO")

    ''' Net '''
    Net = DDSPNet().float()
    Net = nn.DataParallel(Net,device_ids=[0,1])
    Net = Net.to(DEVICE)
    
    lossNET = MSSLoss(FFT_SIZES)
    lossNET = nn.DataParallel(lossNET,device_ids=[0,1])
    lossNET=lossNET.to(DEVICE) 
    ''' Data '''
    Dataset = Dataset()
    Dataloader = DataLoader(Dataset,batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADER)

    VALDataset = DatasetVAL()
    VALDataloader = DataLoader(VALDataset,batch_size=BATCH_SIZE_VAL,
                            shuffle=SHUFFLE_DATALOADER)   
    ''' Train '''
    train(Net,lossNET, Dataloader,VALDataloader,NUMBER_EPOCHS, PRINT_LEVEL)
