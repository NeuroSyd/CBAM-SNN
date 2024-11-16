# a better version of model constructor
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
from snntorch import surrogate
import torch
import torch.nn as nn
import math

# config model style: 
'''
CNN layer/ SNN layer: name outchannel kernel stride padding skipfrom skipto residual
Linear layer: name 
Reisidual layer: same as CNN layer
'''


class DepthwiseSeparableConv(nn.Module):
    '''
    basic convolution block
    args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        stride: stride
        padding: padding
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class Tmaxpool(nn.Module):
    def __init__(self,kernel_size,stride,padding) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        x = x.transpose(0,1)
        timerange = x.shape[0]
        rec = []
        for step in range(timerange):
            rec.append(self.mp(x[step]))
        x = torch.stack(rec)
        x = x.transpose(0,1)
        return x
class Cnnbase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, normalize=nn.BatchNorm2d, activation=nn.ReLU):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            padding (int): Padding of the convolution (default: 0)
            normalize (nn.Module): Normalization layer (default: nn.BatchNorm2d)
            activation (nn.Module): Activation function (default: nn.ReLU)
        """
        super(Cnnbase, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        if isinstance(normalize, nn.BatchNorm2d):
            self.bn = normalize(out_channels)
        else:
            self.bn = None
        self.relu = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x
class TCnnbase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,time_step=4, normalize=snn.BatchNormTT2d, activation=nn.ReLU6):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            padding (int): Padding of the convolution (default: 0)
            time_step (int): Number of time steps (default: 4)
            normalize (nn.Module or None): Normalization layer (default: snn.BatchNormTT2d)
            activation (nn.Module or None): Activation function (default: nn.ReLU6)
        """
        super(TCnnbase, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        if normalize is not None:
            self.bn = normalize(out_channels,time_steps=time_step)
        else:
            self.bn = nn.Identity()
        if activation is not None:
            self.relu = activation(inplace=True)
        else:
            self.relu = nn.Identity()
        self.time_step = time_step

    def forward(self, x):
        out = []
        x = x.transpose(0,1)
        for i in range(self.time_step):
            #print(i)
            x0 = x[i]
            bni = self.bn[i]
            x0  = self.conv(x0)
            x0  = bni(x0 )
            out.append(self.relu(x0 ))
        out = torch.stack(out)
        #print(out.size())
        out = out.transpose(0,1)
        #print(out.size())
        return out
class LCBV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, time_steps, padding,activation = nn.SiLU):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            time_steps (int): Number of time steps (default: 4)
            padding (int): Padding of the convolution (default: 0)
            activation (nn.Module or None): Activation function (default: nn.ReLU6)
        """
        super().__init__()
        self.time_steps = time_steps
        self.outchannels = out_channels
        self.stride = stride


        self.B1 = torch.rand(1)
        self.TH = torch.rand(1)
        self.gF =  torch.rand(1)
        self.LIF = snn.Leaky(beta=self.B1, threshold=self.TH, learn_beta=True, learn_threshold=True, 
                            
                             graded_spikes_factor=self.gF,learn_graded_spikes_factor=True)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )
        if activation is not None:
            self.act1 = activation(inplace=True) 
        else:
            self.act1 = nn.Identity()
        
        self.inv_LIF = snn.Leaky(beta=self.B1, threshold=self.TH, learn_beta=True, learn_threshold=True, 
                             
                             graded_spikes_factor=self.gF,learn_graded_spikes_factor=True)
        self.convM = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
        )
        if activation is not None:
            self.act2 = activation(inplace=True) 
        else:
            self.act2 = nn.Identity()
        
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.BN = snn.BatchNormTT2d(out_channels, time_steps=time_steps)
    def forward(self, x):
        x = x.transpose(0, 1)
        timerange, batch_size, _, height, width = x.shape
        mem = self.LIF.init_leaky()
        mem_inv = self.inv_LIF.init_leaky()
        final_spk_rec = torch.zeros((timerange, batch_size, self.outchannels, height // self.stride, width // self.stride), device=x.device)
        for steps in range(timerange):
            invsteps = timerange - steps - 1
            bn = self.BN[steps]
            bn_inv = self.BN[invsteps]
            x0, mem = self.LIF(x[steps], mem)
            x0 = self.conv(x0)
            x0 = self.act1(x0)
            spk = bn(x0)
            xinv, mem_inv = self.inv_LIF(x[invsteps], mem_inv)
            xinv = self.conv(xinv)
            xinv = self.act2(xinv)
            spk_inv = bn_inv(xinv)
            final_spk_rec[steps] += spk*0.5
            final_spk_rec[invsteps] += spk_inv*0.5
        x = final_spk_rec.transpose(0, 1)
        return x
class LCBV2Small(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, time_steps, padding,activation = nn.SiLU):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            time_steps (int): Number of time steps (default: 4)
            padding (int): Padding of the convolution (default: 0)
            activation (nn.Module or None): Activation function (default: nn.SiLU)
        """
        
        super().__init__()
        self.timesteps = time_steps
        self.outchannels = out_channels
        self.stride = stride


        self.B1 = torch.rand(1)
        self.TH = torch.rand(1)
        self.gF =  torch.rand(1)
        self.LIF = snn.Leaky(beta=self.B1, threshold=self.TH, learn_beta=True, learn_threshold=True, 
                             graded_spikes_factor=self.gF,learn_graded_spikes_factor=True)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.BN = snn.BatchNormTT2d(out_channels,time_steps)
    def forward(self, x):
        x = x.transpose(0, 1)
        timerange, batch_size, _, height, width = x.shape
        mem = self.LIF.init_leaky()
        final_spk_rec = torch.zeros((timerange, batch_size, self.outchannels, height // self.stride, width // self.stride), device=x.device)
        for steps in range(timerange):
            bn = self.BN[steps]
            x0, mem = self.LIF(x[steps], mem)
            x0 = self.conv(x0)
            spk = bn(x0)
            final_spk_rec[steps] += spk
        x = final_spk_rec.transpose(0, 1)

        return x
    
    
class MSM2(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride,padding,time_steps,v2f = True,activation = nn.SiLU,activation2 = nn.SiLU):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            time_steps (int): Number of time steps (default: 4)
            padding (int): Padding of the convolution (default: 0)
            v2f (bool): Whether to use V2F or not (default: True)
            activation (nn.Module or None): Activation function for the first layer (default: nn.SiLU)
            activation2 (nn.Module or None): Activation function for the following residual layer (default: nn.SiLU)
        """
        super().__init__()
        assert in_channels<out_channels
        print(time_steps)
        if v2f:
            self.L = nn.ModuleList(
                [
                    LCBV2(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2(in_channels=out_channels//2,out_channels=out_channels,kernel_size=kernel_size,stride=1,
                          time_steps=time_steps,padding=padding,activation=activation),
                ]
            )
            self.RLCB = TCnnbase(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding=0,time_step=time_steps,)
            
        else:
            self.L = nn.ModuleList(
                [
                    LCBV2Small(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2Small(in_channels=out_channels//2,out_channels=out_channels,kernel_size=kernel_size,stride=1,
                          time_steps=time_steps,padding=padding,activation=activation),
                ]
            )
            self.RLCB = TCnnbase(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding=0,time_step=time_steps,)
            
        self.Ms = MS(in_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,
                     padding=padding,v2f = v2f,activation=activation2)
        self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        Xl = x
        Xr = x
        for layer in self.L:
            Xl = layer(Xl)
        Xr = self.RLCB(Xr)#batch,time,c,w,h
        x = Xl + Xr
        x = self.Ms(x)
        return x

class TCFATestBlockVCSTPN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, timesteps, activation=nn.SiLU):
        """
        Paper used CBAM combine modula: only channel attention and spatial attention
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution (default: 1)
            padding (int): Padding of the convolution (default: 0)
            timesteps (int): Number of time steps (default: 4)
            activation (nn.Module or None): Activation function (default: nn.SiLU)
        """
        super().__init__()
        self.timesteps = timesteps
        print(f'time_steps:{self.timesteps}')
        self.out_channels = out_channels
        self.stride = stride
        self.B1 = torch.rand(1)
        self.TH = torch.rand(1)
        self.gF = torch.rand(1)
        self.factor = nn.Parameter(torch.ones(1))
        self.LIF = snn.Leaky(beta=self.B1, threshold=self.TH, learn_beta=True, learn_threshold=True, graded_spikes_factor=self.gF, learn_graded_spikes_factor=True)
        self.B2 = torch.rand(1)
        self.TH2 = torch.rand(1)
        self.gF2 = torch.rand(1)
        self.factor2 = nn.Parameter(torch.ones(1))
        self.LIF2 = snn.Leaky(beta=self.B2, threshold=self.TH2, learn_beta=True, learn_threshold=True, graded_spikes_factor=self.gF2, learn_graded_spikes_factor=True)

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=out_channels//2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        self.convM = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=out_channels//2,
                kernel_size=1,
                stride=stride,
                padding=0
            ),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=out_channels//2,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )

        self.BN = snn.BatchNormTT2d(out_channels//2, self.timesteps)
        self.BN2 = snn.BatchNormTT2d(out_channels, self.timesteps)
        self.Clinear = nn.Sequential(
            nn.Linear(out_channels// 2, out_channels // 4),
            nn.ReLU6(inplace=True),
            nn.BatchNorm1d(out_channels // 4),
            nn.Linear(out_channels // 4, out_channels//2),
        )
        self.Tlinear = nn.Sequential(
            nn.Linear(self.timesteps, self.timesteps // 2),
            nn.BatchNorm1d(self.timesteps // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(self.timesteps // 2, self.timesteps),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                LCBV2(in_channels, out_channels, kernel_size=1, stride=1, time_steps=timesteps,padding=0,activation=activation),
            )
        else:
            self.residual = None

    def forward(self, x):
        x = x.transpose(0, 1)
        time_steps, batch_size, channels, height, width = x.shape
        mem = self.LIF.init_leaky()
        mem2 = self.LIF2.init_leaky()
        mem2
        xout = []
        mem_rec = []
        if self.residual is not None:
            x0 = self.residual(x.transpose(0, 1)).transpose(0, 1)
        else:
            x0 = x
    
        for step in range(self.timesteps):
            bn = self.BN[step]
            xo = x[step]  
            xo, mem = self.LIF(xo, mem)
            xo = self.conv(xo)
            xo = bn(xo)
            # Channel Attention
            channel_att = torch.mean(xo, dim=[2, 3])  
            channel_att_max, _ = torch.max(xo, dim=2)  
            channel_att_max, _ = torch.max(channel_att_max, dim=2)  
            channel_att = self.Clinear(channel_att)  
            channel_att_max = self.Clinear(channel_att_max)
            channel_att += channel_att_max
            channel_att = torch.sigmoid_(channel_att).unsqueeze(-1).unsqueeze(-1) 
            xo = xo * channel_att
            # Spatial Attention
            avg_pool = torch.mean(xo, dim=1)  
            max_pool, _ = torch.max(xo, dim=1)
            spatial_att = torch.stack([avg_pool, max_pool], dim=1)  
            spatial_att = self.spatial_conv(spatial_att)  
            spatial_att = torch.sigmoid_(spatial_att)
            xo = xo * spatial_att
            xout.append(xo)
            mem_rec.append(self.convM(mem))
        x = torch.stack(xout)  
        xout = []
        for step in range(self.timesteps):
            bn = self.BN2[step]
            xo = x[step]  
            xo, mem = self.LIF2(xo, mem2)
            xo = self.conv2(xo)
            xo = bn(xo)
            xout.append(xo)
        x = torch.stack(xout)
        x = x+x0
        x = x.transpose(0, 1)
        return x

class SnnResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            time_steps (int): Number of time steps
            use_residual (bool): Whether to use residual connection (default: True)
            num_repeats (int): Number of repeats of the residual block (default: 1)
            activation (nn.Module): Activation function (default: nn.SiLU)
        """
        super(SnnResidualBlock,self).__init__()
        self.layers = nn.ModuleList()
        self.residual = nn.Identity()
        for _ in range(num_repeats):
                self.layers += [
                    nn.Sequential(
                        LCBV2Small(in_channels=in_channels,out_channels=out_channels//2,kernel_size=3,padding=1,stride=1,time_steps=time_steps,activation=activation),
                        LCBV2Small(out_channels//2,out_channels,kernel_size = 3,padding = 1,stride=1,time_steps=time_steps,activation=activation),
                    )
                ]

        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.residual = nn.Sequential(
                LCBV2Small(in_channels, out_channels, kernel_size=1, stride=1, time_steps=time_steps,padding=0,activation=activation),
            )
        self.num_repeats = num_repeats
    def forward(self,x):
        
        for layer in self.layers:
            if self.use_residual:
                x0 = x
                x0 = self.residual(x0)
                x = layer(x) + x0
            else:
                x = layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,times=4):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(times):
            self.layers+=[
                nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels//2, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU6(inplace=True),
                DepthwiseSeparableConv(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)),
                nn.ReLU6(inplace=True),]
    def forward(self, x):

        for layer in self.layers:
            x = x + layer(x)
        
        return x

class MLPLayer(nn.Module):
    def __init__(self, indim, outdim, normalize=nn.BatchNorm1d, activation=nn.ReLU):
        """
        Args:
            indim (int): Input dimension
            outdim (int): Output dimension
            normalize (nn.Module or None): Normalization layer (default: nn.BatchNorm1d)
            activation (nn.Module or None): Activation function (default: nn.ReLU)
        """
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(indim, outdim)
        if isinstance(normalize, nn.BatchNorm1d):
            self.bn = normalize(outdim)
        else:
            self.bn = None
        if activation is not None:
            self.act = activation(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
class SnnMLPLayer(nn.Module):
    def __init__(self,indim, outdim, time_steps,normalize=snn.BatchNormTT1d, activation=nn.SiLU,outF = False):
        """
        Args:
            indim (int): Input dimension
            outdim (int): Output dimension
            time_steps (int): Number of time steps
            normalize (nn.Module or None): Normalization layer (default: snn.BatchNormTT1d)
            activation (nn.Module or None): Activation function (default: nn.SiLU)
            outF (bool): Whether to output the final state (default: False)
        """
        super(SnnMLPLayer, self).__init__()
        self.linear = nn.ModuleList()
        for _ in range(time_steps):
            self.linear += [nn.Linear(indim, outdim)]
        self.B1 = torch.rand(1)
        self.TH = torch.rand(1)
        self.outF = outF
        self.LIF = snn.Leaky(beta=self.B1,threshold=self.TH,learn_beta=True,learn_threshold=True)
        if normalize is not None:
            self.bn = normalize(outdim,time_steps)
        else:
            self.bn = nn.ModuleList()
            for i in range(time_steps):
                self.bn += [nn.Identity(outdim)]
        if activation is not None:
            self.act = activation(inplace=True)
        else:
            self.act = nn.Identity()
    def forward(self, x):

        mem = self.LIF.init_leaky()
        x = x.transpose(0, 1)
        spk_rec = []
        for st in range(x.shape[0]):
            xo = x[st]
            bni = self.bn[st]
            linear_T = self.linear[st]
            xo,mem = self.LIF(xo,mem)
            if self.bn is not None:
                xo= bni(linear_T(xo))
            else:
                xo = linear_T(xo)
            xo = self.act(xo)
            spk_rec.append(xo)
        if self.outF:
            x = torch.stack(spk_rec)
        else:
            x = torch.stack(spk_rec)
            x = x.transpose(0, 1)
        #print(x.shape)
        return x
class TFaltten(nn.Module):
    def __init__(self):
        """
        Initialize the TFaltten layer.

        This layer will flatten the input tensor to a 2D tensor, but it will
        preserve the time dimension.
        """
        super(TFaltten, self).__init__()
        self.flatten = nn.Flatten()
    def forward(self,x):
        x = x.transpose(0,1)
        #print(x.shape)
        output = []
        for i in range(x.shape[0]):
            x0 = x[i]
            x0 = self.flatten(x0)
            output.append(x0)
        x = torch.stack(output)
        x = x.transpose(0,1)
        #print(x.shape)
        return x

class MS(nn.Module):
    def __init__(self,in_channels,kernel_size,stride ,time_steps,padding,v2f = True,activation=nn.SiLU):
        """
        Initialize the MS layer.

        This layer will create two sub-layers. The first layer is a LCBV2 layer with
        half of the output channels. The second layer is also a LCBV2 layer with the
        remaining half of the output channels. The two sub-layers are connected in
        series.

        Args:
            in_channels (int): Number of input channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution
            time_steps (int): Number of time steps
            padding (int): Padding of the convolution
            v2f (bool): Whether to use V2F or not (default: True)
            activation (nn.Module or None): Activation function for the two sub-layers (default: nn.SiLU)
        """
        super().__init__()
        self.time_steps = time_steps
        if v2f:
            self.LCB1 = LCBV2(in_channels=in_channels,out_channels=in_channels//2,
                              kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding,activation=activation)
            self.LCB2 = LCBV2(in_channels=in_channels//2,out_channels=in_channels,
                              kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding,activation=activation)
        else:
            self.LCB1 = LCBV2Small(in_channels=in_channels,out_channels=in_channels//2,kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding)
            self.LCB2 = LCBV2Small(in_channels=in_channels//2,out_channels=in_channels,kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding)
    def forward(self,x):
        xo = x
        x = self.LCB1(x)
        x = self.LCB2(x)
        x += xo
        return x

class SNNBlockV3M1(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,time_steps,v2f = True,activation = nn.SiLU):
        """
        Initialize the SNNBlockV3M1 module.(EMS M1 residual layer) 

        This module consists of two branches, L and R, each with a sequence of layers. The choice of layers
        is determined by the `v2f` flag. Each branch processes the input data through distinct convolutional
        and pooling layers.
        """
        super().__init__()
        assert in_channels >= out_channels
        if v2f:
            self.L = nn.ModuleList(
                [
                    LCBV2(in_channels=in_channels,out_channels=out_channels//2,
                          kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2(in_channels=out_channels//2,out_channels=out_channels,
                          kernel_size=3,stride=1,time_steps=time_steps,padding=1,activation=activation),
                ]
            )
            self.R = nn.ModuleList(
                [
                    Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding),
                    LCBV2(in_channels=in_channels,out_channels=out_channels,
                          kernel_size=3,stride=1,time_steps=time_steps,padding=1,activation=activation),
                ]
            )
        else:
            self.L = nn.ModuleList(
                [
                    LCBV2(in_channels=in_channels,out_channels=out_channels//2,
                          kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2(in_channels=out_channels//2,out_channels=out_channels,
                          kernel_size=3,stride=1,time_steps=time_steps,padding=1,activation=activation),
                ]
            )
            self.R = nn.ModuleList(
                [
                    Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding),
                    LCBV2(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,time_steps=time_steps,padding=1,activation=activation),
                ]
            )
        self.Ms = MS(in_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding,v2f = v2f)
    def forward(self,x):
        xL = x
        xR = x
        for layer in self.L:
            xL = layer(xL)
        for layer in self.R:
            xR = layer(xR)
        x = xL+xR
        x = self.Ms(x)
        return x
    
class SNNBlockVS2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,time_steps,v2f = True,activation = nn.SiLU,activation2 = nn.SiLU):
        """
        Initialize the SNNBlockVS2 layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution
            time_steps (int): Number of time steps
            padding (int): Padding of the convolution
            v2f (bool): Whether to use V2F or not (default: True)
            activation (nn.Module or None): Activation function for the two sub-layers (default: nn.SiLU)
            activation2 (nn.Module or None): Activation function for the following residual layer (default: nn.SiLU)
        """
        super().__init__()
        assert in_channels<out_channels
        R_C = out_channels - in_channels
        if v2f:
            self.L = nn.ModuleList(
                [
                    LCBV2(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2(in_channels=out_channels//2,out_channels=out_channels,kernel_size=kernel_size,stride=1,
                          time_steps=time_steps,padding=padding,activation=activation),
                ]
            )
            self.RLCB = LCBV2(in_channels=in_channels,out_channels=R_C,kernel_size=kernel_size,stride=1,
                              time_steps=time_steps,padding=1,activation=activation)
        else:
            self.L = nn.ModuleList(
                [
                    LCBV2Small(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding),
                    LCBV2Small(in_channels=out_channels//2,out_channels=out_channels,
                               kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding),
                ]
            )
            self.RLCB = LCBV2Small(in_channels=in_channels,out_channels=R_C,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=1)
        self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        Xl = x
        Xr = x
        for layer in self.L:
            Xl = layer(Xl)
        Xr = self.mp(Xr)
        Xr0 = Xr
        Xr0 = self.RLCB(Xr0)
        Xr = torch.concat([Xr,Xr0],dim=2)
        x = Xl + Xr
        return x


    
class SNNBlockV3M2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,time_steps,v2f = True,activation = nn.SiLU,activation2 = nn.SiLU):
        """
        Initialize the EMS-M2 residual layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution
            time_steps (int): Number of time steps
            padding (int): Padding of the convolution
            v2f (bool): Whether to use V2F or not (default: True)
            activation (nn.Module or None): Activation function for the two sub-layers (default: nn.SiLU)
            activation2 (nn.Module or None): Activation function for the following residual layer (default: nn.SiLU)
        """
        super().__init__()
        assert in_channels<out_channels
        R_C = out_channels - in_channels
        if v2f:
            self.L = nn.ModuleList(
                [
                    LCBV2(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding,activation=activation),
                    LCBV2(in_channels=out_channels//2,out_channels=out_channels,kernel_size=kernel_size,stride=1,
                          time_steps=time_steps,padding=padding,activation=activation),
                ]
            )
            self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
            self.RLCB = LCBV2(in_channels=in_channels,out_channels=R_C,kernel_size=kernel_size,stride=1,
                              time_steps=time_steps,padding=1,activation=activation)
        else:
            self.L = nn.ModuleList(
                [
                    LCBV2Small(in_channels=in_channels,out_channels=out_channels//2,
                        kernel_size=kernel_size,stride=stride,
                        time_steps=time_steps,padding=padding),
                    LCBV2Small(in_channels=out_channels//2,out_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding),
                ]
            )
            self.RLCB = LCBV2Small(in_channels=in_channels,out_channels=R_C,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=1)
        self.Ms = MS(in_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,
                     padding=padding,v2f = v2f,activation=activation2)
        self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        Xl = x
        Xr = x
        for layer in self.L:
            Xl = layer(Xl)
        Xr = self.mp(Xr)
        Xr0 = Xr
        Xr0 = self.RLCB(Xr0)#batch,time,c,w,h
        Xr = torch.concat([Xr,Xr0],dim=2)#b,t,c,w,h
        x = Xl + Xr
        x = self.Ms(x)
        return x

class Full_E_RES(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,timesteps=4,v2f = True,activation = nn.SiLU,):
        """
        Initialize the Full spike skip connect adapter.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            stride (int): Stride of the convolution
            padding (int): Padding of the convolution
            timesteps (int): Number of time steps (default: 4)
            v2f (bool): Whether to use V2F or not (default: True)
            activation (nn.Module or None): Activation function (default: nn.SiLU)
        """
        super().__init__()
        self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
        self.RLCB = LCBV2Small(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,
                              time_steps=timesteps,padding=padding,activation=activation)
    def forward(self,x):
        #print(x.shape)
        x = self.mp(x)
        x = self.RLCB(x)#batch,time,c,w,h
        return x
    



# define the funtions of all used modual
ConvList = [nn.Conv1d, nn.Conv2d, nn.Conv3d,Cnnbase,ResidualBlock,LCBV2,SNNBlockV3M2,SNNBlockV3M1,TCnnbase,MSM2,TCFATestBlockVCSTPN,SNNBlockVS2]
LinearList = [nn.Linear,MLPLayer,SnnMLPLayer]
ResidualList = [ResidualBlock,SnnResidualBlock]
class ClassificationModel_New_New(nn.Module):
    def __init__(self, num_classes, in_channels=3, config_set=None, snnF=False,imageshape=(3, 224, 224),timestpes = 4):
        """
        Initialize the ClassificationModel_New_New.

        Args:
            num_classes (int): Number of output classes
            in_channels (int): Number of input channels (default: 3)
            config_set (list or None): List of tuples containing the configuration of the layers (default: None)
            snnF (bool): Whether to use SNN or not (default: False)
            imageshape (tuple): Shape of the input image (default: (3, 224, 224))
            timestpes (int): Number of time steps (default: 4)
        """
        super(ClassificationModel_New_New, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.config_set = config_set
        self.shape = imageshape
        self.layers = nn.ModuleList()
        self.skip_convs = nn.ModuleDict()  
        self.skip_connections = {} 
        self.snnFlag = snnF
        self.timestpes = timestpes
        self._create_conv_layers()
        
        

    def _autoshape(self, in_channels, H, W):
        """
        Calculate the size of the flatten layer before the fully connected layer.

        Args:
            in_channels (int): Number of input channels
            H (int): Height of the feature map
            W (int): Width of the feature map

        Returns:
            int: The size of the flatten layer
        """
        return in_channels * H * W
        

    def _create_conv_layers(self):
        """
        Create convolutional layers based on the configuration set (example is provided in following code).

        Attributes:
            in_channels (int): Initial number of input channels.
            shape (tuple): Tuple containing the initial height and width of input images.
            config_set (list): List of tuples specifying layer configurations.
            snnFlag (bool): Flag indicating whether to use SNN configurations.
            timestpes (int): Number of time steps for SNN layers.
            skip_convs (nn.ModuleDict): Dictionary for storing 1x1 convolutions for 
                skip connection adjustments.
            skip_connections (dict): Dictionary for storing skip connection information.
        """
        in_channels = self.in_channels
        h, w = self.shape[1], self.shape[2]  
        channels_at_layer = {}  
        shape_log = []

        for idx, config in enumerate(self.config_set):
            layer_class = config[0]
            layer_params = config[1]
            skip_layers = config[2] if len(config) > 2 else []  
            skip_type = config[3] if len(config) > 3 else None  
            

            in_channels_for_layer = in_channels
            if skip_layers and skip_type == "concat":
                # adapt in_channels_for_layer to skip layers (concat)
                adjusted_channels = 0
                for skip_layer_idx in skip_layers:
                    ori_shape = shape_log[skip_layer_idx]
                    skip_out_channels = channels_at_layer[skip_layer_idx]
                    if layer_class in ConvList:
                        h_n,w_n = shape_log[-1]
                        print(f'conv layer: skip {skip_layer_idx} to {idx} with h {h_n},w{w_n}')
                    elif layer_class in ResidualList:
                            h_n,w_n = shape_log[-1]
                            print(f'residual layer: skip {skip_layer_idx} to {idx} with h {h_n},w{w_n}')
                    # if the output channels of the skip layer are not equal to the input channels of the current layer
                    if skip_out_channels != in_channels or ori_shape[0] != h_n or ori_shape[1] != w_n:
                        adjusted_channels += in_channels
                        conv_layer_name = f"conv_{skip_layer_idx}_to_{idx}"
                        ori_shape = shape_log[skip_layer_idx]
                        print('create transform layer for concat connet')
                        if ori_shape[0] != h_n or ori_shape[1] != w_n:
                            print(f'ori_shape: {ori_shape} h: {h} w: {w}')
                            rate = int(ori_shape[0]/h_n)
                            print(f'rate: {rate}')
                        else:
                            rate =1
                        # shape adjustment 
                        if self.snnFlag:
                                print('snnflag')
                                print(f'ori_shape: {ori_shape} h: {h} w: {w},tps:{self.timestpes}')
                                self.skip_convs[conv_layer_name] = TCnnbase(skip_out_channels, in_channels, kernel_size=1,stride=rate,padding=0,time_step=self.timestpes, normalize=snn.BatchNormTT2d, activation=nn.SiLU)
                        else:
                            self.skip_convs[conv_layer_name] = nn.Conv2d(skip_out_channels, in_channels, kernel_size=1,stride=rate)
                    else:
                        adjusted_channels += skip_out_channels
                # final in_channels for current layer
                in_channels_for_layer = in_channels + adjusted_channels
            elif skip_layers and skip_type == "add":
                # adapt in_channels_for_layer to skip layers (add)
                for skip_layer_idx in skip_layers:
                    ori_shape = shape_log[skip_layer_idx]
                    skip_out_channels = channels_at_layer[skip_layer_idx]
                    if layer_class in ConvList:
                        h_n,w_n = shape_log[-1]
                        print(f'conv layer: skip {skip_layer_idx} to {idx} with h {h_n},w{w_n}')
                    elif layer_class in ResidualList:
                            h_n,w_n = shape_log[-1]
                            print(f'residual layer: skip {skip_layer_idx} to {idx} with h {h_n},w{w_n}')
                    # if not the same, the in_channel of the current layer should be adjusted
                    if skip_out_channels != in_channels or ori_shape[0] != h_n or ori_shape[1] != w_n:
                        # create transform layer
                        conv_layer_name = f"conv_{skip_layer_idx}_to_{idx}"
                        print('create transform layer for add connet')
                        if ori_shape[0] != h_n or ori_shape[1] != w_n:
                            print(f'ori_shape: {ori_shape} h: {h_n} w: {w_n}')
                            rate = int(ori_shape[0]/h_n)
                            print(f'rate: {rate}')
                        else:
                            rate =1
                        if self.snnFlag:
                                # for add skip connection
                                self.skip_convs[conv_layer_name] = Full_E_RES(skip_out_channels, in_channels, kernel_size=1,stride=rate,padding=0,timesteps=self.timestpes, activation=nn.SiLU)
                        else:
                            # for add skip connection
                            self.skip_convs[conv_layer_name] = nn.Conv2d(skip_out_channels, in_channels, kernel_size=1,stride=rate)

                    in_channels_for_layer = in_channels
            else:
                in_channels_for_layer = in_channels  # no skip layers

            if layer_class in LinearList:
                # if previous layer is a linear layer, we need to add one flatten layer before next layer
                if isinstance(self.layers[-1], tuple(ConvList)) or isinstance(self.layers[-1], tuple(ResidualList)):
                    print(f'linear part:')
                    if layer_class == SnnMLPLayer:
                        self.layers.append(TFaltten())
                        print('Tflatten')
                    else:
                        self.layers.append(nn.Flatten())
                        print('flatten')
                    in_channels = self._autoshape(in_channels, h, w)
                    h, w = 1, 1  # full connected layer don't have height and width
                print(f'linear {layer_class} id: {in_channels} od: {layer_params[0]}')
                layer = layer_class(in_channels, *layer_params)
                in_channels = layer_params[0]  # update in_channels
            elif layer_class in ResidualList:
                # deal with residual block
                out_channels = layer_params[0]
                if self.snnFlag:
                    stride = layer_params[3]
                else:
                    stride = layer_params[1]

                in_channels = layer_params[0]  # update in_channels
                shape_log.append((h,w))
                print(f"Residual Block: in_channels={in_channels_for_layer}, out_channels={out_channels}, repeat={stride}")
                layer = layer_class(in_channels_for_layer, *layer_params)
                in_channels = out_channels  # update in_channels
            else:
                # deal with conv layer
                print(f'conv {layer_class} K: {layer_params[1]},S: {layer_params[2]},P: {layer_params[3]}')
                print(f'in_C: {in_channels_for_layer},out_C: {layer_params[0]}')
                layer = layer_class(in_channels_for_layer, *layer_params)
                kernel_size, stride, padding = layer_params[1], layer_params[2], layer_params[3]
                h = ((h + 2 * padding - kernel_size) // stride ) + 1
                w = ((w + 2 * padding - kernel_size) // stride ) + 1
                in_channels = layer_params[0]  # update in_channels
                shape_log.append((h,w))
                print((h,w))
            # record the output channels of the current layer
            channels_at_layer[idx] = in_channels
            self.layers.append(layer)
            # record skip connection information if skip_layers is not empty
            if skip_layers:
                self.skip_connections[idx] = {'skip_layers': skip_layers, 'skip_type': skip_type, 'in_channels': in_channels_for_layer}
            #print(len(shape_log))

    def forward(self, x):
        outputs = {}  # storage all the output for each layer
        last_feature = None  # storage the output of the last layer
        has_skip_connections = len(self.skip_connections) > 0
        for idx, layer in enumerate(self.layers):
            
            if has_skip_connections and idx in self.skip_connections:
                skip_info = self.skip_connections[idx]
                skip_layers = skip_info['skip_layers']
                skip_type = skip_info['skip_type']
                in_channels_for_layer = skip_info['in_channels']

                if skip_type == "add":
                    skip_input = x  # add input 
                elif skip_type == "concat":
                    skip_input = [x]  # concat input

                # process input data layers with skip connections
                for skip_layer_idx in skip_layers:
                    skip_output = outputs[skip_layer_idx]
                    if self.snnFlag:
                        if skip_output.size(2) != x.size(2):
                            conv_layer_name = f"conv_{skip_layer_idx}_to_{idx}"
                            skip_output = self.skip_convs[conv_layer_name](skip_output)
                    else:
                        if skip_output.size(1) != in_channels_for_layer:
                            conv_layer_name = f"conv_{skip_layer_idx}_to_{idx}"
                            skip_output = self.skip_convs[conv_layer_name](skip_output)

                    if skip_type == "add":
                        skip_input = skip_input + skip_output  # add
                    elif skip_type == "concat":
                        skip_input.append(skip_output)  # concat 
                    
                if skip_type == "concat":
                    dim = 2 if len(skip_input[0].shape) > 4 else 1
                    x = torch.cat(skip_input, dim=dim)
                elif skip_type == "add":
                    x = skip_input  

            x = layer(x)  
            outputs[idx] = x  # storage the output

            if idx == len(self.layers) - 2:  # umap output 
                last_feature = x

        return x, last_feature  # final output and UMAP feature 

class DeelpCnn8(nn.Module):
    # the model base on the paper: 
    def __init__(self, in_channels, num_classes):
        super(DeelpCnn8, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(100352, 1000),
            nn.Dropout(0.5),
        )
        self.outfc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        features = x
        x = self.outfc(x)
        return x, features

class BaseLineModel(nn.Module):
    # one adapted cnn model for classification
    def __init__(self, config_set=[
        'ResNet18'], in_channels=3, num_classes=1000):
        super(BaseLineModel, self).__init__()
        # create baseline model
        base_model = config_set[0]
        print(f'base model: {base_model}')
        if base_model == 'ResNet18':
            self.model = models.resnet18(pretrained=False)
        elif base_model == 'ResNet34':
            self.model = models.resnet34(pretrained=False)
        elif base_model == 'ResNet50':
            self.model = models.resnet50(pretrained=False)
        elif base_model == 'ResNet101':
            self.model = models.resnet101(pretrained=False)
        elif base_model == 'ResNet152':
            self.model = models.resnet152(pretrained=False)
        else:
            raise ValueError("Unsupported ResNet model. Choose from 'resnet18', 'resnet34', or 'resnet50'.")

        # setting for input layer 
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=self.model.conv1.out_channels,  
                kernel_size=self.model.conv1.kernel_size,    
                stride=self.model.conv1.stride,              
                padding=self.model.conv1.padding,            
                bias=self.model.conv1.bias is not None      
            )
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        for _, layer in list(self.model.named_children())[:-1]:
            x = layer(x)
        features = x.view(x.size(0), -1)
        outputs = self.model.fc(features)
        return outputs, features 
class CustomInceptionV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(CustomInceptionV3, self).__init__()
        # loading inception v3 model
        self.model = models.inception_v3(pretrained=False)
        if in_channels != 3:
            self.model.Conv2d_1a_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.model.Conv2d_1a_3x3.out_channels,
                kernel_size=self.model.Conv2d_1a_3x3.kernel_size,
                stride=self.model.Conv2d_1a_3x3.stride,
                padding=self.model.Conv2d_1a_3x3.padding,
                bias=self.model.Conv2d_1a_3x3.bias is not None
            )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        if self.training and self.model.aux_logits:
            aux = self.model.AuxLogits(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features = x.clone()
        x = self.model.fc(x)
        return x, features
    
'''
Test and demo
'''    
if __name__ == "__main__":
    CNN_config_set=[]

    # example config_set
    start_dim = 4
    time_steps = 1
    v2f = False
    rn = 8
    SNN_config_set =  [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, None), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, None), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [3],"concat"),
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, None), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (SnnResidualBlock,( start_dim * 4, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, None), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 8, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            
            (SNNBlockVS2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, None), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,nn.SiLU,False), [], None),
            (SnnMLPLayer,(10,time_steps,snn.BatchNormTT1d,nn.SiLU,True), [], None),
        ]
# Model Test
    x = torch.randn(16 ,3, 224, 224)
    model = DeelpCnn8(in_channels=3, num_classes=10)
    output,f = model(x)
    print(f'out: {output.shape}')
