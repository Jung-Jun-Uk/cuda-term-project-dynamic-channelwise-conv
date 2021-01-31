import torch
import torch.nn as nn
from torch.nn import functional as F
from ._conv_cuda.conv import Conv

def CustomConv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                 stride=1, dilation=1, is_bias=True, custom_conv=True):
    if custom_conv:
        return Conv(in_channels, out_channels, kernel_size, padding, stride, dilation, is_bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size, 
                     padding=padding, stride=stride, dilation=dilation, bias=is_bias)
    

class LeNetpp(nn.Module):    
    def __init__(self, num_classes, custom_conv=True):
        super(LeNetpp, self).__init__()
        self.conv1_1 = CustomConv2d(1, 32, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = CustomConv2d(32, 32, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = CustomConv2d(32, 64, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = CustomConv2d(64, 64, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = CustomConv2d(64, 128, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = CustomConv2d(128, 128, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu3_2 = nn.PReLU()

        self.conv4_1 = CustomConv2d(128, 128, 5, stride=1, padding=2, custom_conv=custom_conv)
        self.prelu4_1 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*3*3,10)
        self.prelu_fc1 = nn.PReLU()
        
    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))

        x = self.prelu4_1(self.conv4_1(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        
        return x