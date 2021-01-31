import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('./')
from models.adaptive_conv_cuda.dwconv import AdaptiveDepthwiseConv2dFunction

def make_tuple(value, n_value):
    if not isinstance(value, (list, tuple)):
        return (value,) * n_value

    else:
        n_item = len(value)

        if n_item > n_value:
            raise ValueError(
                f'Number items does not match with requirements: {n_item}, expected: {n_value}'
            )

        if len(value) == n_value:
            return value

        return value * n_value

class AdaptiveDepthWiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size, inference, num_classes=3,
                padding=0, stride=1, dilation=1, bias=False):
        super(AdaptiveDepthWiseConv2d, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.inference = inference

        gain = nn.init.calculate_gain('relu')
        he_std = gain * (channels * kernel_size ** 2) ** (-0.5)  # He init
        
        self.candidate_weight = nn.Parameter(
            torch.randn(num_classes, channels, 1, kernel_size, kernel_size) * he_std
        )
        
        self.conv1 = nn.Conv2d(channels, channels * num_classes, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.inference:
            self.stride = make_tuple(stride, 2)
            self.padding = make_tuple(padding, 2)
        
    def forward(self, x):
        B, C, _, _ = x.size()

        weight_prob = self.avg_pool(self.conv1(x))
        weight_prob = weight_prob.view(B, C, self.num_classes, 1, 1)
        weight_prob = F.softmax(weight_prob, dim=2)
        print(weight_prob.size())
        if self.inference:
            #weight_prob = weight_prob.permute(0,2,1,3,4).unsqueeze(3)
            c_weight = self.candidate_weight.unsqueeze(0)
            
            c_weight = c_weight.permute(0,2,1,3,4,5)
            weight_prob = weight_prob.unsqueeze(3)
            weight = (c_weight * weight_prob).sum(dim=2)
            weight = weight.squeeze(2)
            print(weight.size())
            return AdaptiveDepthwiseConv2dFunction.apply(x, weight, self.stride, self.padding)

        out = []
        for i in range(self.num_classes):
            o = F.conv2d(x, self.candidate_weight[i], stride=self.stride, 
                           padding=self.padding, groups=C)
            out.append(o.unsqueeze(2))
        out = torch.cat(out, dim=2)
        out = (out * weight_prob).sum(dim=2)
        return out
    
if __name__ == "__main__":
    adaptive_conv2d = AdaptiveDepthWiseConv2d(channels=64, kernel_size=3, inference=True, num_classes=4, padding=1, stride=1).cuda()
    x = torch.randn(64,64,224,224).cuda()
    out = adaptive_conv2d(x)
    print(out.size())