#Big ups to https://www.youtube.com/watch?v=IHq1t7NxS8k for model walkthrough

import torch
import torch.nn as nn
import torchvision.transforms as tf

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels=in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)



class UNET(nn.Module):
    def __init__(
        self,in_channels = 3,out_channels = 1,features = [64,128,256,512],
    ):
        super(UNET,self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)
        

        #Down part of UNET

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #UP part of UNET

        for feature in features:




