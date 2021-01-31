import os
import sys
import errno
import os.path as osp
import numpy as np

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')
            
    def __del__(self):
        self.close()
    
    def __exit__(self, *args):
        self.close()
    
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
    
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
            
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_model(model, epoch, name, save_dir):
    dirname = osp.join(save_dir, 'weights')
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, name + 'epoch_' + str(epoch+1) + '.pth')
    torch.save(model.state_dict(), save_name)
