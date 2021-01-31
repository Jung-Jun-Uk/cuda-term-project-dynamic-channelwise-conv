'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_dwconv.dwconv import DepthwiseConv2d
from .adaptive_dwconv import AdaptiveDepthWiseConv2d

def CustomDepthwiseConv2d(in_channels, out_channels, groups=None, kernel_size=3, padding=1, 
                 stride=1, bias=False, custom=True):
    if custom:
        #return DepthwiseConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
        #                       padding=padding, stride=stride, bias=False)
        return AdaptiveDepthWiseConv2d(channels=in_channels, kernel_size=kernel_size, inference=False, num_classes=9, padding=padding, stride=stride)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                     groups=in_channels, stride=stride, padding=padding, bias=False)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, custom):
        super(Block, self).__init__()
        self.stride = stride
        
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CustomDepthwiseConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, 
                                           groups=planes, bias=False, custom=custom)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    """ cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)] """

    cfg = [(1,  16, 1, 1),
           (6,  24, 1, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 1, 2),
           (6,  64, 1, 2),
           (6,  96, 1, 1),
           (6, 160, 1, 2),
           (6, 320, 1, 1)]           

    def __init__(self, num_classes=10, custom=True):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.custom = custom
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.custom))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2().cuda()
    x = torch.randn(2,3,32,32).cuda()
    y = net(x)
    print(y.size())

# test()