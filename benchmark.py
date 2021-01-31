import argparse
import math
import time

import torch
from models.mobilenetv2 import MobileNetV2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-s', '--state-size', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

X = torch.randn(1,3,32,32).cuda()
model = MobileNetV2(num_classes=10, custom=False).cuda()

forward_min = math.inf
forward_time = 0

for _ in range(options.runs):

    start = time.time()
    y = model(X)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale    
forward_average = forward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {2}'.format(
    forward_min, forward_average, options.scale))