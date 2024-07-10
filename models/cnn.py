import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        r"""
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 

        output = F.avg_pool2d(x, kernel_size=(2, 2))
        
        return output 


class Cnn(nn.Module):
    def __init__(self, classes_num):
        super(Cnn, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=0.,
            f_max=8000,
            n_mels=128,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock(in_channels=1, out_channels=32)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=256)

        self.onset_fc = nn.Linear(256, classes_num)

    def forward(self, audio):
        r"""
        Args:
            audio: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, classes_num)
        """
        
        x = self.mel_extractor(audio)
        # shape: (B, 1, F, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b c f t -> b c t f')
        # shape: (B, 1, T, F)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # shape: (B, C, T, F)

        x, _ = torch.max(x, dim=-1)
        x, _ = torch.max(x, dim=-1)

        output = torch.sigmoid(self.onset_fc(x))

        return output