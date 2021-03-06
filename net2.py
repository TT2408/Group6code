# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 00:02:41 2021

@author: 82520
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as func

from parameters import *
import pdb
class ConvNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(ConvNet, self).__init__()

        """ 1st step """
        self.conv1=nn.Conv1d(input_size, output_size, 1, stride=1)
        self.norm1 = nn.LayerNorm(output_size)
        self.relu1=nn.LeakyReLU()
        """ 2nd step """
        self.conv2=nn.Conv1d(output_size, output_size, 1, stride=1)
        self.norm2 = nn.LayerNorm(output_size)
        self.relu2=nn.LeakyReLU()
        """ 3rd step """
        self.conv3=nn.Conv1d(output_size, output_size, 1, stride=1)
        self.norm3 = nn.LayerNorm(output_size)
        self.relu3=nn.LeakyReLU()
        
    def forward(self, x):
        y = self.relu1(self.norm1(self.conv1(x.permute(0,2,1)).permute(0,2,1)))
        y = self.relu2(self.norm2(self.conv2(y.permute(0,2,1)).permute(0,2,1)))
        y = self.relu3(self.norm3(self.conv3(y.permute(0,2,1)).permute(0,2,1)))

        return y
    
    
class DDSPNet(nn.Module):
    def __init__(self):
        super(DDSPNet, self).__init__()
        self.con0 = ConvNet(Binsize,HIDDEN_DIM)
        self.con1 = ConvNet(Binsize,HIDDEN_DIM)
        self.con2 = ConvNet(HIDDEN_DIM,HIDDEN_DIM)
        self.gru = nn.GRU(2* HIDDEN_DIM+Binsize, HIDDEN_DIM, batch_first=True)
        self.dense_additive = nn.Linear(HIDDEN_DIM, NUMBER_HARMONICS+1)
        #self.celu_additive = nn.CELU()

        if NOISE_AMPLITUDE:
            self.dense_noise = nn.Linear(HIDDEN_DIM, NUMBER_NOISE_BANDS+1)
        else:
            self.dense_noise = nn.Linear(HIDDEN_DIM, NUMBER_NOISE_BANDS)

    def forward(self, x_mel):    
        
        y_mel0=self.con0(x_mel)          
        y_mel1=self.con1(x_mel)        
        y=torch.cat((x_mel,y_mel0, y_mel1), dim=2)
        y = self.gru(y)[0]   #nn,gru return (output, hn)   
        y=self.con2(y) 
        
        y_noise = self.dense_noise(y)             
        y_additive = self.dense_additive(y)           
        
        if FINAL_SIGMOID == "None":
            pass
        elif FINAL_SIGMOID == "Usual":
            y_additive = torch.sigmoid(y_additive)
            y_noise = torch.sigmoid(y_noise)
        elif FINAL_SIGMOID == "Scaled":
            y_additive = 2 * torch.sigmoid(y_additive)**np.log(10) + 10e-7
            y_noise = 2 * torch.sigmoid(y_noise)**np.log(10) + 10e-7
        elif FINAL_SIGMOID == "Mixed":
            y_additive = 2 * torch.sigmoid(y_additive) ** np.log(10) + 10e-7
            y_noise = torch.sigmoid(y_noise)
        else:
            raise AssertionError("Final sigmoid not understood.")

        return y_additive, y_noise
