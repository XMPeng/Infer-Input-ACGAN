"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import pickle
import time
# import random
# import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
# from utils import weights_init, compute_acc
# from network_Xiaoming import _netG, _netD
from model_64_infer import *
# from network_Xiaoming import _netG, _netD
from vgg_16 import VGG16
# from folder import ImageFolder

from random import randrange

from dataload_Xiaoming_dis import Places9

from scipy.optimize import basinhopping

## Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 根据实际来设置
print('Device:', DEVICE)


## arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nz', type=int, default=108, help='size of the latent z vector')
# parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--load_pretrain', action='store_true', help='load pretrain parameters')
# parser.add_argument('--useNoise', action='store_true', help='add a little noise to training data')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--folder_GD', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--folder_VGG', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=8, help='Number of classes for AC-GAN')
# parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

# parser.add_argument('--exDir', default='.', help='folder to output inverting results')

# parser.add_argument('--load_val_list', action='store_true', help='load pretrain parameters')

opt = parser.parse_args()
print(opt)

nz = int(opt.nz)
num_classes = int(opt.num_classes)

## load models
netG = Generator(nz)
netG = netG.to(DEVICE)
netG.load_state_dict(torch.load('./models/netG_epoch.pth'))

netG.fc_1 = nn.Linear(8192, 100)
netG.fc_1.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_1.bias.data = netG.fc1.weight[:, 0].squeeze()+netG.fc1.bias.data

netG.fc_2 = nn.Linear(8192, 100)
netG.fc_2.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_2.bias.data = netG.fc1.weight[:, 1].squeeze()+netG.fc1.bias.data

netG.fc_3 = nn.Linear(8192, 100)
netG.fc_3.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_3.bias.data = netG.fc1.weight[:, 2].squeeze()+netG.fc1.bias.data

netG.fc_4 = nn.Linear(8192, 100)
netG.fc_4.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_4.bias.data = netG.fc1.weight[:, 3].squeeze()+netG.fc1.bias.data

netG.fc_5 = nn.Linear(8192, 100)
netG.fc_5.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_5.bias.data = netG.fc1.weight[:, 4].squeeze()+netG.fc1.bias.data

netG.fc_6 = nn.Linear(8192, 100)
netG.fc_6.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_6.bias.data = netG.fc1.weight[:, 5].squeeze()+netG.fc1.bias.data

netG.fc_7 = nn.Linear(8192, 100)
netG.fc_7.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_7.bias.data = netG.fc1.weight[:, 6].squeeze()+netG.fc1.bias.data

netG.fc_8 = nn.Linear(8192, 100)
netG.fc_8.weight.data = netG.fc1.weight[:, num_classes:]
netG.fc_8.bias.data = netG.fc1.weight[:, 7].squeeze()+netG.fc1.bias.data

torch.save(netG.state_dict(), './models/netG_epoch_ext.pth')






