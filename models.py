import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
import datasets.labels as l
from utils import convertColour
from datasets.cityscapes import setupDatasetsAndLoaders

# Define Model 
class Resnet_LSTM(nn.Module):
    def __init__(self, num_classes=19):
        # Init from nn.Module
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
#       self.vgg16 = models.vgg16(pretrained=False)
        self.classifier32 = nn.Sequential(nn.Conv2d(512, 512, 1),
                                         nn.Conv2d(512,num_classes, 1))
        self.classifier16 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                         nn.Conv2d(256,num_classes, 1))
        self.classifier8 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                         nn.Conv2d(128,num_classes, 1))
        self.deconv = nn.ConvTranspose2d(num_classes*2, num_classes, 32, stride=32)
        self.deconv32 = nn.ConvTranspose2d(num_classes*2, num_classes, 4, stride=2, padding=1)
        self.deconv16 = nn.ConvTranspose2d(num_classes*2, num_classes, 4, stride=2, padding=1)
        self.deconv8 = nn.ConvTranspose2d(num_classes*2, num_classes, 8, stride=8)
        self.lstm = nn.LSTM(num_classes*7*14, num_classes*7*14)
        self.flatten = nn.Flatten()
        

    def forward(self,x):
        res = []
        for ind,x1 in enumerate(x):
            x1 = self.resnet18.conv1(x1)
            x1 = self.resnet18.bn1(x1)
            x1 = self.resnet18.relu(x1)
            x1 = self.resnet18.maxpool(x1)
            x1 = self.resnet18.layer1(x1)
            x1 = self.resnet18.layer2(x1)
            if ind == 3:
                x4_8 = self.classifier8(x1)
                x4_8 = x4_8
            x1 = self.resnet18.layer3(x1)
            if ind == 3:
                x4_16 = self.classifier16(x1)
                x4_16 = x4_16
            x1 = self.resnet18.layer4(x1)
            x1 = self.classifier32(x1)
            if ind == 3:
                x4_32 = x1
            x1 = self.flatten(x1)
            res.append(x1)
        x = torch.stack(res,1)
        x, _ = self.lstm(x)
        
        x = x[:,3,:].reshape(-1, 19, 7, 14)
        x = torch.cat([x,x4_32],1)
        x = self.deconv32(x)
        x = torch.cat([x,x4_16],1)
        x = self.deconv16(x)
        x = torch.cat([x,x4_8],1)
        x = self.deconv8(x)

        return x
    
class PSP(nn.Module):
    def __init__(self, num_classes=19,pool_scales=(1,2,3,6)):
        super().__init__()
        
        #PSP variables
        self.pool_scale = pool_scales
        dim = 16

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(num_classes, dim, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_last = nn.Sequential(
            nn.Conv2d(num_classes+len(pool_scales)*dim, num_classes, kernel_size=1)
        )
        
    def forward(self,x):    
        
        #Referred from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/e21bUqcHCfmkmkE1f9HU3LuU3pozaYeUsZ6b71a/models.py
#         print("After Autoencoder:",x.shape)
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            pool = pool_scale(x)
#             print("After pooling:",pool.shape)
            p_layer = nn.functional.interpolate(
                pool,
                (input_size[2], input_size[3]),
                mode='bilinear')
#             print("After upsampling:",p_layer.shape)
            psp_out.append(p_layer)            
        x = torch.cat(psp_out, 1)
#         print("After concatenation:",x.shape)
        x = self.conv_last(x)
#         print("After last convolution:",x.shape)
        
        return x
        
class BigDLModel(nn.Module):
    def __init__(self, use_psp =False,num_classes=19,pool_scales=(1,2,3,6)):
        super().__init__()
        self.use_psp = use_psp
      
        self.resnet_lstm = Resnet_LSTM(num_classes)
        if use_psp:
            self.psp = PSP(num_classes,pool_scales)
            
    def forward(self,x):
        x = self.resnet_lstm(x)
        if self.use_psp:
            x = self.psp(x)
        
        return x
        
class Resnet_FCN(nn.Module):
    def __init__(self, num_classes=19):
        # Init from nn.Module
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
#       self.vgg16 = models.vgg16(pretrained=False)
        self.classifier32 = nn.Sequential(nn.Conv2d(512, 512, 1),
                                         nn.Conv2d(512,num_classes, 1))
        self.classifier16 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                         nn.Conv2d(256,num_classes, 1))
        self.classifier8 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                         nn.Conv2d(128,num_classes, 1))
#         self.deconv = nn.ConvTranspose2d(num_classes*2, num_classes, 32, stride=32)
        self.deconv32 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.deconv16 = nn.ConvTranspose2d(num_classes*2, num_classes, 4, stride=2, padding=1)
        self.deconv8 = nn.ConvTranspose2d(num_classes*2, num_classes, 8, stride=8)      

    def forward(self,x):
        x1 = x[-1]
        x1 = self.resnet18.conv1(x1)
        x1 = self.resnet18.bn1(x1)
        x1 = self.resnet18.relu(x1)
        x1 = self.resnet18.maxpool(x1)
        x1 = self.resnet18.layer1(x1)
        x1 = self.resnet18.layer2(x1)

        x4_8 = self.classifier8(x1)
        x4_8 = x4_8
        
        x1 = self.resnet18.layer3(x1)
        
        x4_16 = self.classifier16(x1)
        x4_16 = x4_16
        
        x1 = self.resnet18.layer4(x1)
        x1 = self.classifier32(x1)

        x = self.deconv32(x1)
        x = torch.cat([x,x4_16],1)
        x = self.deconv16(x)
        x = torch.cat([x,x4_8],1)
        x = self.deconv8(x)

        return x
    
class NoLSTMModel(nn.Module):
    def __init__(self, use_psp =False,num_classes=19,pool_scales=(1,2,3,6)):
        super().__init__()
        self.use_psp = use_psp
      
        self.resnet_fcn = Resnet_FCN(num_classes)
        if use_psp:
            self.psp = PSP(num_classes,pool_scales)
            
    def forward(self,x):
        x = self.resnet_fcn(x)
        if self.use_psp:
            x = self.psp(x)
        
        return x
    
        