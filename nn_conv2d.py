#四个尺度，例如s/2，其它三个尺度经过上采样首先把图片尺度回复与s/2一样，batch_size不变，然后用方法只将通道数相加，最后除以4，最后得到与s/2相同的尺寸的图片
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d


class dz(nn.Module):
    def __init__(self):
        super(dz, self).__init__()

        self.conv1=Conv2d(in_channels=3,out_channels=3,kernel_size=1,padding=0,stride=1,bias=True)
    def forward(self, x):
        self.conv1(x)
        rgb_down42=self.conv1(x)(F.interpolate(rgb_down22,scale_factor=2,mode='bilinear'))
        rgb_down82 = self.conv1(x)(F.interpolate(rgb_down42, scale_factor=2, mode='bilinear'))
        rgb_down162 = self.conv1(x)(F.interpolate(rgb_down82, scale_factor=2, mode='bilinear'))





