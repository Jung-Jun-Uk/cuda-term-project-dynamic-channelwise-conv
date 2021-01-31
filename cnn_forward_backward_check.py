import numpy as np
import torch
from torch import nn
from models._conv_cuda.conv import Conv, ConvFunction
from models.torch_dwconv.dwconv import DepthwiseConv2d, DepthwiseConv2dFunction
from models.adaptive_dwconv import AdaptiveDepthWiseConv2d, AdaptiveDepthwiseConv2dFunction

from torch.autograd import gradcheck

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def check_equal(first, second, verbose=False):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i), rtol=1e-6)

def check_cnn():
    test_data = torch.randn([3, 3, 3, 3]).cuda()

    conv = Conv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    conv_torch = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    W = nn.Parameter(torch.randn(conv.weight.size()))
    b = nn.Parameter(torch.randn(conv.bias.size()))

    print("Weights and bias size : ", W.size(), b.size())
    conv.weight = W
    conv.bias = b
    conv_torch.weight = W
    conv_torch.bias = b

    conv = conv.cuda()
    conv_torch = conv_torch.cuda()

    first = conv(test_data)
    second = conv_torch(test_data)
    check_equal(first,second)

    if gradcheck(ConvFunction.apply, [test_data, conv.weight, conv.bias, conv.params], eps=1e-2, atol=1e-2, raise_exception=True):
        print('Ok')

def check_depthwise_convolution(N, C, H, W, kernel_size, stride, padding):

    input = torch.randn(N,C,H,W).cuda()

    depthwise_conv = DepthwiseConv2d(in_channels=C, out_channels=C, 
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False).cuda()

    
    pytorch_dwconv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=kernel_size, groups=C, stride=stride, padding=padding, bias=False).cuda()
    pytorch_dwconv.weight = depthwise_conv.weight
    
    custom = depthwise_conv(input)
    native = pytorch_dwconv(input) 
    check_equal(custom, native)

    if gradcheck(DepthwiseConv2dFunction.apply, [input, depthwise_conv.weight, depthwise_conv.stride, depthwise_conv.padding], eps=1e-2, atol=1e-2, raise_exception=True):
        print('Ok')
    else:
        print("Not matched !!", N, C, H, W, kernel_size, stride, padding)

def check_adaptive_depthwise_convolution(N, C, H, W, kernel_size, stride, padding):

    input = torch.randn(N,C,H,W).cuda()

    adaptive_depthwise_conv = AdaptiveDepthWiseConv2d(channels=C, kernel_size=kernel_size, 
                                        inference=False, num_classes=9, padding=1, stride=1).cuda()
    adaptive_depthwise_conv_infer = AdaptiveDepthWiseConv2d(channels=C, kernel_size=kernel_size, 
                                        inference=True, num_classes=9, padding=1, stride=1).cuda()
    
    adaptive_depthwise_conv_infer.candidate_weight = adaptive_depthwise_conv.candidate_weight

    train_out = adaptive_depthwise_conv(input)
    infer_out = adaptive_depthwise_conv_infer(input) 

    print(train_out.size(), infer_out.size())
    check_equal(train_out, infer_out)
    
if __name__ == "__main__":    
    BATCH_SIZE = 32
    test_nchw_kernel_size_stride_padding = [
        [BATCH_SIZE, 8, 34, 34, 3, 1, 1],
        [BATCH_SIZE, 8, 34, 34, 5, 1, 2],
        [BATCH_SIZE, 8, 34, 34, 3, 2, 1],
        [BATCH_SIZE, 8, 34, 34, 5, 2, 2],
        [BATCH_SIZE, 8, 34, 34, 3, 1, 0],
        [BATCH_SIZE, 8, 34, 34, 5, 1, 0],
        [BATCH_SIZE, 8, 34, 34, 3, 2, 0],
        [BATCH_SIZE, 8, 34, 34, 5, 2, 0],
        [BATCH_SIZE, 8, 33, 35, 3, 1, 1],
        [BATCH_SIZE, 8, 33, 35, 5, 1, 2],
        [BATCH_SIZE, 8, 33, 35, 3, 2, 1],
        [BATCH_SIZE, 8, 33, 35, 5, 2, 2],
        [BATCH_SIZE, 8, 34, 34, 3, 1, 1],
        [BATCH_SIZE, 8, 34, 34, 5, 1, 2],
        [BATCH_SIZE, 8, 34, 34, 3, 2, 1],
        [BATCH_SIZE, 8, 34, 34, 5, 2, 2],
        [BATCH_SIZE, 8, 34, 34, 3, 1, 0],
        [BATCH_SIZE, 8, 34, 34, 5, 1, 0],
        [BATCH_SIZE, 8, 34, 34, 3, 2, 0],
        [BATCH_SIZE, 8, 34, 34, 5, 2, 0],
        [BATCH_SIZE, 8, 33, 35, 3, 1, 1],
        [BATCH_SIZE, 8, 33, 35, 5, 1, 2],
        [BATCH_SIZE, 8, 33, 35, 3, 2, 1],
        [BATCH_SIZE, 8, 33, 35, 5, 2, 2],
        [BATCH_SIZE, 8, 16, 16, 3, 1, 1],
        [BATCH_SIZE, 8, 16, 16, 5, 1, 2],
        [BATCH_SIZE, 8, 16, 16, 3, 1, 1],
        [BATCH_SIZE, 8, 16, 16, 5, 1, 2]
    ]        
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    print(os.environ)
    #for info in test_nchw_kernel_size_stride_padding:
    #    N, C, H, W, kernel_size, stride, padding = info
        #check_depthwise_convolution(N, C, H, W, kernel_size, stride, padding)
    check_adaptive_depthwise_convolution(32, 3, 16, 16, 3, 1, 1)