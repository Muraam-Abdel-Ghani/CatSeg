# import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn.functional as F
from pyconvresnet import pyconvresnet152, pyconvresnet50, pyconvresnet101
# from .pyconvresnet import pyconvresnet152, pyconvresnet50, pyconvresnet101
from ptflops import get_model_complexity_info


'''
This script creates the CatSeg model / PFDNet with the new MSIFA module that replaces the Conv layers with Depth-wise separable convolutions.
'''

class PyConvResNet_Separate(nn.Module):
    def __init__(self):
        super(PyConvResNet_Separate,self).__init__()
        
        pyconv_resnet_model = pyconvresnet152(pretrained=True)
        # pyconv_resnet_model = pyconvresnet50(pretrained=True)
        self.Conv1 = nn.Sequential(*list(pyconv_resnet_model.children())[0:4]) 
        self.Conv2 = nn.Sequential(*list(pyconv_resnet_model.children())[4]) 
        self.Conv3 = nn.Sequential(*list(pyconv_resnet_model.children())[5])
        self.Conv4 = nn.Sequential(*list(pyconv_resnet_model.children())[6])

    def forward(self,x):
        # print(x.shape) #    torch.Size([1, 3, 512, 512])
        out1 = self.Conv1(x) 
        # print(f"Conv1 out: {out1.shape}") # Conv1 out: torch.Size([1, 256, 128, 128])
        out2 = self.Conv2(out1)
        # print(f"Conv2 out: {out2.shape}") # Conv2 out: torch.Size([1, 512, 64, 64])
        out3 = self.Conv3(out2)
        # print(f"Conv3 out: {out3.shape}") # Conv3 out: torch.Size([1, 1024, 32, 32])
        out4 = self.Conv4(out3)
        # print(f"Conv3 out: {out4.shape}") # Conv3 out: torch.Size([1, 2048, 16, 16])

        return out1, out2, out3, out4

    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # depthwise: one filter per input channel
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # pointwise: 1x1 convolution
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
# Multi-Scale Insturment Feature Attention Module
class MSIFA_DW(nn.Module): 
    def __init__(self, in_channels, embed_dim=512):
        super(MSIFA_DW, self).__init__()
        self.Conv5x5 = DepthwiseSeparableConv(in_channels=in_channels, out_channels=embed_dim, kernel_size=5, padding=2, stride=1, dilation=1)
        
        # Scale 1 branch
        self.Conv1x11 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 11), stride=1, 
                                  padding=(0, 5)  # Padding for width maintaining
                        )
        self.Conv11x1 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(11, 1), stride=1, 
                                  padding=(5, 0)  # Padding for height maintaining
                        )
        
        # Scale 2 branch
        self.Conv1x21 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 21), stride=1, 
                                  padding=(0, 10)  # Padding for width maintaining
                        )
        self.Conv21x1 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(21, 1), stride=1, 
                                  padding=(10, 0)  # Padding for height maintaining
                        )
        # Scale 3 branch
        self.Conv1x31 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 31), stride=1, 
                                  padding=(0, 15)  # Padding for width maintaining
                        )
        self.Conv31x1 = DepthwiseSeparableConv(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(31, 1), stride=1, 
                                  padding=(15, 0)  # Padding for height maintaining
                        )
        
        self.Conv1x1 = nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=(1,1), stride=1, padding=0, dilation=1)
        
        
    def forward(self, features):
        
        conv_5x5_features = self.Conv5x5(features) # [b, c, w, h] -> [b, 512, w, h]
        
        scale_1_11 = self.Conv1x11(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]
        scale_11_1 = self.Conv11x1(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]

        # Residual addition
        residual_scale_11 = scale_1_11 + scale_11_1 # -> [b, 512, w, h]
        
        
        scale_1_21 = self.Conv1x21(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]
        scale_21_1 = self.Conv21x1(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]

        # Residual addition
        residual_scale_21 = scale_1_21 + scale_21_1 # -> [b, 512, w, h]
        
        scale_1_31 = self.Conv1x31(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]
        scale_31_1 = self.Conv31x1(conv_5x5_features) # [b, 512, w, h] -> [b, 512, w, h]

        # Residual addition
        residual_scale_31 = scale_1_31 + scale_31_1 # -> [b, 512, w, h]
        
        residual_scales = residual_scale_11 + residual_scale_21 + residual_scale_31
        
        concat_feature_scales= torch.cat((conv_5x5_features, residual_scales), dim=1) # 2x[b,512,w,h] -> [b,1024,w,h]
        
        compressed_feature_scales = self.Conv1x1(concat_feature_scales) # [b, 1024, w, h] -> [b, 512, w, h]
        
        attended_features = torch.mul(conv_5x5_features, compressed_feature_scales) # [b, 512, w, h]
        
        return attended_features
    

class InstrumentDecoder_1(nn.Module):
    def __init__(self):
        super(InstrumentDecoder_1, self).__init__()
        
        self.msifa_1 = MSIFA_DW(in_channels=256, embed_dim=512) # [b, 256, 128, 128] -> [b, 512, 128, 128]
        
    def forward(self, encoder_outputs):
        x1 = encoder_outputs # [b, 256, 128, 128]
        
        attended_x1 = self.msifa_1(x1) # [b,256, 128, 128] -> [b, 512, 128, 128]
        
        return attended_x1
    
class InstrumentDecoder_2(nn.Module):
    def __init__(self):
        super(InstrumentDecoder_2, self).__init__()

        self.msifa_2 = MSIFA_DW(in_channels=512, embed_dim=512) # [b, 512, 64, 64] -> [b, 512, 64, 64]

    def forward(self, encoder_outputs):
        x2 = encoder_outputs # [b, 512, 64, 64]

        attended_x2 = self.msifa_2(x2) # [b, 512, 64, 64] -> [b, 512, 64, 64]

        attended_x2_up = F.interpolate(attended_x2, size=(128, 128), mode='bilinear', align_corners=False) # [b, 512, 64, 64] -> [b, 512, 128,128]

        return attended_x2_up

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=1024):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        batch, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        mlp_x = self.proj(x)
        mlp_x= mlp_x.permute(0,2,1).reshape(batch, -1, h, w)
        return mlp_x
    
class AnatomyDecoder_3(nn.Module):
    def __init__(self):
        super(AnatomyDecoder_3, self).__init__()
        self.mlp3 = MLP(input_dim=1024, embed_dim=512)
        
        
    def forward(self, encoder_outputs):
        
        x3 = encoder_outputs # [batch, 1024, 32, 32]
        
        mlp_x3 = self.mlp3(x3) # [batch, 1024, 32, 32] -> [batch, 512, 32, 32]
        
        
        # print(mlp_x3.shape)
        
        mlp_x3_up_1 = F.interpolate(mlp_x3, size=(64, 64), mode='bilinear', align_corners=False) # [b, 512, 32, 32] -> [b, 512, 64, 64]
        mlp_x3_up_2 = F.interpolate(mlp_x3_up_1, size=(128, 128), mode='bilinear', align_corners=False) # [b, 512, 64, 64] -> [b, 512, 128, 128]
        
    
        return mlp_x3_up_2
    
class AnatomyDecoder_4(nn.Module):
    def __init__(self):
        super(AnatomyDecoder_4, self).__init__()
        self.mlp4_1 = MLP(input_dim=2048, embed_dim = 1024)
        self.mlp4_2 = MLP(input_dim=1024, embed_dim = 512)
        
        
    def forward(self, encoder_outputs):
        
        x4 = encoder_outputs # [batch, 2048, 16, 16]
        
        mlp_x4_1 = self.mlp4_1(x4) # [batch, 2048, 16, 16] -> [batch, 1024, 16, 16]
        mlp_x4_2 = self.mlp4_2(mlp_x4_1) # [batch, 1024, 16, 16] -> [batch, 512, 16, 16]
        
        mlp_x4_up_1 = F.interpolate(mlp_x4_2, size=(32, 32), mode='bilinear', align_corners=False) # [b, 512, 16, 16] -> [b, 512, 32, 32]
        mlp_x4_up_2 = F.interpolate(mlp_x4_up_1, size=(64, 64), mode='bilinear', align_corners=False) # [b, 512, 32, 32] -> [b, 512, 64, 64]
        mlp_x4_up_3 = F.interpolate(mlp_x4_up_2, size=(128, 128), mode='bilinear', align_corners=False) # [b, 512, 64, 64] -> [b, 512, 128,128]

        return mlp_x4_up_3
    

# Parallel Feature Decoding Network              
class PFDNet(nn.Module): 
    def __init__(self, num_classes):
        super(PFDNet, self).__init__()
        self.num_classes = num_classes

        self.Backbone = PyConvResNet_Separate()
        self.anatomy_decoder_x3 = AnatomyDecoder_3()
        self.anatomy_decoder_x4 = AnatomyDecoder_4()
        self.instrument_decoder_x1 = InstrumentDecoder_1()
        self.instrument_decoder_x2 = InstrumentDecoder_2()
         
            
        
        # for compressing channels and increasing feature map size
        
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1), #[b,2048, 128,128] -> [b,1024,256,256]
            nn.BatchNorm2d(1024),
            nn.ReLU())
        
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), #[b,1024, 256,256] -> [b, 512, 512,512]
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        
        self.compress_channels1 = nn.Sequential( # -> [b, 256, 512,512]
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        
        self.compress_channels2 = nn.Sequential(  # -> [b, 128, 512,512]
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        
                
        self.compress_channels3 = nn.Sequential(  # -> [b, 64, 512,512]
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        
                       
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # -> [b, num_classes, 512,512]
        
    def forward(self, x):
        
        out1, out2, out3, out4 = self.Backbone(x)     
        
        mlp_x3 = self.anatomy_decoder_x3(out3)#([b, 1024,32,32], [b, 2048,16,16]) -> ([b, 512,128,128], [b, 512,128,128])
        
        mlp_x4 = self.anatomy_decoder_x4(out4)#([b, 1024,32,32], [b, 2048,16,16]) -> ([b, 512,128,128], [b, 512,128,128])
        
        attended_x1 = self.instrument_decoder_x1(out1) #([b, 256,128,128], [b, 512,64,64]) -> ([b, 512,128,128], [b, 512,128,128])
        attended_x2 = self.instrument_decoder_x2(out2) #([b, 256,128,128], [b, 512,64,64]) -> ([b, 512,128,128], [b, 512,128,128])
        
        concat_features = torch.concat((attended_x1, attended_x2, mlp_x3, mlp_x4), dim=1) # 4x[b,512,128,128] -> [b,2048, 128,128]
        
        
        
        concat_features = self.convTrans1(concat_features)
        concat_features = self.convTrans2(concat_features)
        concat_features = self.compress_channels1(concat_features)
        concat_features = self.compress_channels2(concat_features)
        concat_features = self.compress_channels3(concat_features)
        
        concat_features = self.output_conv(concat_features)
        
              
        return concat_features
if __name__ == '__main__':
    model = PFDNet( num_classes=1)
    device=torch.device("cuda")
    model = model.to(device)


    template = torch.ones((1, 3, 512, 512)).to(device)
    detection= torch.ones((1, 1, 512, 512))
        
    model(template)
    print("done") #torch.Size([1,1,512,512])
    # print(summary(model, (3,512,512)))
    
    flops, params = get_model_complexity_info(model, (3,512,512), as_strings=True, print_per_layer_stat=True)

    print(f"FLOPs: {flops}, Params: {params}")