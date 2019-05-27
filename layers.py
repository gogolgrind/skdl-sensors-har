
"""
    multi-sin-conv is an extention of the https://github.com/mravanelli/SincNet
"""

import torch
import numpy as np
import  torch.nn as nn
import torch.nn.functional as F
import math 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Flip(nn.Module):
    
    def __init__(self,dim):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self,x):
        xsize = x.size()
        dim = x.dim() + dim if self.dim < 0 else self.dim
        x = x.contiguous()
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, torch.arange(x.size(1)-1, -1, -1), :]
        return x.view(xsize)


class Sinc(nn.Module):
    
    def __init__(self,device='cuda'):
        super(self.__class__, self).__init__()
        self.flip = Flip(0)
        self.device = device
        
    def forward(self,band,t_right):
        y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
        y_left = self.flip(y_right)
        y = torch.cat([y_left,torch.ones(1).to(self.device),y_right])
        return y

class LayerNorm(nn.Module):
    """
        https://github.com/mravanelli/SincNet
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class multichannel_sinc_conv(nn.Module):
    """
     extention of the https://github.com/mravanelli/SincNet
    """
    def __freqinit__(self):
        if self.cutfreq_type == 'mel':
            low_freq_mel = 80
            high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
            mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
            f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
            b1=np.roll(f_cos,1)
            b2=np.roll(f_cos,-1)
            b1[0] = 30
            b2[-1] = (self.fs/2)-100
        elif self.cutfreq_type == 'pre_defined':
            b1 = np.array([15,35,80])
            b2 = np.array([25,50,90])
        elif self.cutfreq_type == 'random':
            eeg_const = 100
            b1 = np.random.randint(10,int(self.fs//2)-10,[self.N_filt])
            #b1 = 10 * np.ones([self.N_filt])
            b2 = np.minimum(b1+5,int(self.fs//2))        
            b1 = b1.tolist()
            b2 = b2.tolist()
        else:
            raise NotImplementedError
        bands_low = []
        bands_high = []
        for i in range(self.N_channels):
            bands_low.append(b1)
            bands_high.append(b2)
        bands_low = np.array(bands_low).astype('float32')
        bands_high = np.array(bands_high).astype('float32')
        return (bands_low,bands_high)

    def get_filter_bank(self,beg_freq,end_freq):
        if self.filt_type == 'sinc':
            t_right = (torch.linspace(1, (self.Filt_dim-1)/2, steps=int((self.Filt_dim-1)/2))/self.fs).to(self.device)
            low_pass1 = 2*beg_freq.float()*self.sinc(beg_freq.float()*self.fs,t_right)
            low_pass2 = 2*end_freq.float()*self.sinc(end_freq.float()*self.fs,t_right)
            band_pass = (low_pass2-low_pass1)
            band_pass = band_pass/torch.max(band_pass)
            return band_pass.to(self.device)*self.window
        elif self.filt_type == 'firwin':
            raise NotImplementedError
            cut1 = beg_freq.float().detach().cpu().numpy()*self.fs
            cut2 = end_freq.float().detach().cpu().numpy()*self.fs
            cut1 = np.round(cut1)
            cut2 = np.round(cut2)
            band_pass = sp.signal.firwin(self.Filt_dim, [cut1,cut2], pass_zero=False,fs=self.fs,window='hamming')
            band_pass = torch.from_numpy(band_pass.astype('float32'))
            band_pass = nn.Parameter(band_pass.to(self.device))
            return band_pass
        else:
            raise NotImplementedError

    def __init__(self, 
                 N_filt = 12, 
                 N_channels = 27, 
                 Filt_dim = 5, 
                 fs = 100,
                 cutfreq_type = 'random', 
                 filt_type = 'sinc',
                 device = 'cuda'):
        super(multichannel_sinc_conv,self).__init__()
        self.cutfreq_type = cutfreq_type
        self.fs = float(fs)
        self.device = device
        self.sinc = Sinc(self.device)
        self.filt_type = filt_type
        self.N_channels = N_channels
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        bands = self.__freqinit__()
        self.bands_low = bands[0]
        self.bands_high = bands[1]
        self.filt_low = nn.Parameter(torch.from_numpy(self.bands_low/self.fs).to(self.device))
        self.filt_band = nn.Parameter(torch.from_numpy((self.bands_high-self.bands_low)/self.fs).to(self.device))        
        n = torch.linspace(0, self.Filt_dim, steps=self.Filt_dim)
        #hamming = 0.54-0.46*torch.cos(2*math.pi*n/self.Filt_dim)
        blackman = 0.42 - 0.5*torch.cos(2*math.pi*n/self.Filt_dim)
        blackman += 0.08*torch.cos(4*math.pi*n/self.Filt_dim)
        self.window = blackman.float().to(self.device)
        
    def forward(self, x):
        k = 0
        min_freq=1.0;
        min_band=10.0;
        filters = torch.zeros((self.N_filt*self.N_channels,self.Filt_dim)).to(self.device)
        for j in range(self.N_channels):    
            filt_beg_freq = torch.abs(self.filt_low)[j] + min_freq/self.fs
            filt_end_freq = (filt_beg_freq+(torch.abs(self.filt_band)))[j]
            for i in range(self.N_filt):
                band_pass = self.get_filter_bank(filt_beg_freq[i],filt_end_freq[i]) 
                filters[k,:] = band_pass
                k+=1
        filters = filters.view(self.N_filt*self.N_channels,1,self.Filt_dim)
        return F.conv1d(x,filters,groups=self.N_channels)