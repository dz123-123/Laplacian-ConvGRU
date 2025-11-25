import torch
import numpy as np
from model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import math
# -*- coding: utf-8 -*-
import argparse
import time
import csv
import math
import numpy as np
import os

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import cv2

import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from datasets_list import MyDataset, Transformer

from itertools import cycle
from tqdm import tqdm
import imageio
import imageio.core.util
import itertools
from path import Path

import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from trainer import validate
from model import *
parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir',type=str,  default='KITTI_LDRN_ResNext101_epoch25/epoch_23_loss_0.2125_2.pkl')
parser.add_argument('--trainfile_kitti', type=str, default = "./datasets/train_all.txt")
parser.add_argument('--testfile_kitti', type=str, default = "./datasets/val_all.txt")
parser.add_argument('--trainfile_nyu', type=str, default = "./datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "./datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "./datasets/KITTI")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=6, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "KITTI")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH', help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--height', type=int, default = 352)
parser.add_argument('--width', type=int, default = 704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Evaluation setting
parser.add_argument('--evaluate',default=True, type=bool, help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)





args = parser.parse_args()
cudnn.benchmark = True
args.max_depth = 80.0
print('=> loading model..')
Model = MRSU(args)
Model = Model.cuda(0)
Model.load_state_dict(torch.load(args.model_dir))
Model.eval()
print("Model Initialized")

test_set = MyDataset(args, train=False)
print("=> Data height: {}, width: {} ".format(args.height, args.width))
print('=> test  samples_num: {}  '.format(len(test_set)))

test_sampler = None

val_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=test_sampler)

cudnn.benchmark = True

if args.dataset == 'KITTI':
    errors, error_names = validate(args, val_loader, Model, logger=None,dataset='KITTI')
    print(errors,error_names)
