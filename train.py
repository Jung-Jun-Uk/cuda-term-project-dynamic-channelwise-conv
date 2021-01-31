import os
import sys
import datetime

import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from data.mnist import MNIST
from data.cifar10 import CIFAR10
from models.lenet import LeNetpp
from models.mobilenetv2 import MobileNetV2
from utils import Logger, AverageMeter, save_model
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def print_config(args):
    conf = vars(args)
    print("Config FILE")
    for key, value in conf.items():
        if key == 'model':
            continue
        print('{:<25} = {}'.format(key,value))
    print("\n\n")

def training(args):
    args.use_gpu = torch.cuda.is_available()
    #os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_devices
    torch.manual_seed('1')

    args.save_dir = osp.join(args.save_dir, args.data, args.model, str(args.custom_conv) +'tiny')
    print(args.save_dir)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + '.txt'))
    print_config(args)

    if args.use_gpu:
        print("Currently using GPU: {}".format(args.cuda_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all('0')
    else:
        print("Currently using CPU")
    print("Creating dataset: {}".format(args.data))

    if args.data == 'mnist':
        dataset = MNIST(args.batch_size, args.use_gpu, args.workers)
    elif args.data == 'cifar10':
        dataset = CIFAR10(args.batch_size, args.use_gpu, args.workers)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    
    args.num_classes = dataset.num_classes

    if args.model == 'lenetpp':
        model = LeNetpp(args.num_classes, custom_conv=args.custom_conv)
    elif args.model == 'mobilenetv2':
        model = MobileNetV2(num_classes=args.num_classes, custom=args.custom_conv)
    print("Creating model: {}".format(args.model))
    #model.load_state_dict(torch.load('work_dir/adaptive_3/cifar10/mobilenetv2/True1/weights/LeNet_epoch_100.pth'))
    if args.use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=5e-04, momentum=0.9)
    
    if args.stepsize > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    acc, err = test(args, model, trainloader, 0)
    print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))

    start_time = time.time()    
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(args, model, criterion, optimizer, trainloader, epoch)
        
        if args.stepsize > 0: 
            scheduler.step()
        
        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Train")
            acc, err = test(args, model, trainloader, epoch)
            print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            print("==> Test")
            acc, err = test(args, model, testloader, epoch)
            print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            save_model(model, epoch, name='LeNet_', save_dir=args.save_dir)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(args, model, criterion, optimizer, trainloader, epoch):
    model.train()
    losses = AverageMeter()
    
    for i, (data, labels) in enumerate(trainloader):
        if args.use_gpu:
            data, labels = data.cuda(), labels.cuda()
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), labels.size(0))
                 
        if (i+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                .format(i+1, len(trainloader), losses.val, losses.avg))

def test(args, model, testloader, epoch):
    model.eval()
    correct, total = 0, 0
                
    with torch.no_grad():
        for data, labels in testloader:
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err
             
def parser():    
    parser = argparse.ArgumentParser(description='Mnist')
    parser.add_argument('--lr'               , default=0.1)
    parser.add_argument('--workers'          , default=4)
    parser.add_argument('--batch_size'       , default=256)
    parser.add_argument('--max_epoch'        , default=100)
    parser.add_argument('--stepsize'         , default=30)
    parser.add_argument('--gamma'            , default=0.1)
    parser.add_argument('--eval_freq'        , default=10)
    parser.add_argument('--print_freq'       , default=50)
    parser.add_argument('--num_classes'      , default=10)
    parser.add_argument('--data'             , choices=['mnist', 'cifar10'])
    parser.add_argument('--model'            , choices=['lenetpp', 'mobilenetv2'])
    parser.add_argument('--custom_conv'      , action='store_true', help='Use custom conv or not')
    parser.add_argument('--save_dir'         , default='work_dir/', help='Resume head path for retraining')
    parser.add_argument('-cuda,', '--cuda_devices', default="1", type=str, help='CUDA_VISIBLE_DEVICES, ex) -cuda "0,1,2,3" ')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    training(parser())
    
