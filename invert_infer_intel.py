"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import statistics
import pickle
import time
import matplotlib.pyplot as plt
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
from model_64_infer3 import *
# from network_Xiaoming import _netG, _netD
from vgg_16 import VGG16
# from folder import ImageFolder

from random import randrange

from dataload_intel import Intel6

from scipy.optimize import basinhopping

# categories = {'highway': 1, 'train_railway': 5}
# categories_list = ['highway', 'train_railway']
# categories = {'fountain': 0, 'highway': 1, 'iceberg': 2, 'ocean': 3, 'river': 4,
#               'train_railway': 5, 'volcano': 6, 'wind_farm': 7}
# categories_list = ['fountain', 'highway', 'iceberg', 'ocean', 'river',
#                    'train_railway', 'volcano', 'wind_farm']
## Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 根据实际来设置
print('Device:', DEVICE)


## arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nz', type=int, default=106, help='size of the latent z vector')
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
parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for AC-GAN')
# parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

# parser.add_argument('--exDir', default='.', help='folder to output inverting results')

# parser.add_argument('--load_val_list', action='store_true', help='load pretrain parameters')

opt = parser.parse_args()
print(opt)

nz = int(opt.nz)
num_classes = int(opt.num_classes)

## load models
netE = Encoder(nz, True)
netG = Generator(nz)
netD = Discriminator(nz, num_classes, 1)

netE = netE.to(DEVICE)
netG = netG.to(DEVICE)
netD = netD.to(DEVICE)

netE.load_state_dict(torch.load('./models_infer_intel5/netE_epoch.pth'))
netG.load_state_dict(torch.load('./models_infer_intel5/netG_epoch.pth'))
netD.load_state_dict(torch.load('./models_infer_intel5/netD_epoch.pth'))

# netE.load_state_dict(torch.load('./models_intel_pixel/netE_epoch.pth'))
# netG.load_state_dict(torch.load('./models_intel_pixel/netG_epoch.pth'))
# netD.load_state_dict(torch.load('./models_intel_pixel/netD_epoch.pth'))


# # specify the gpu id if using only 1 gpu
# if opt.cuda:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id) # 设置使用的GPU，标准操作

# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass

# exDir = os.path.join(opt.exDir, 'inversionExperiments')
# try:
#     os.mkdir(exDir)
# except:
#     print('already exists')

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed) # 生成同一个随机数？？
#
# torch.manual_seed(opt.manualSeed) # 为了让每次的结果一致，对CPU设置随机种子

##############################暂时不用##########################################################
# if opt.cuda:
#     torch.cuda.manual_seed_all(opt.manualSeed) # 对GPU设置随机种子

# # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# cudnn.benchmark = True

# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# some hyper parameters
# ngpu = int(opt.ngpu)


# Define the generator and initialize the weights

def find_single(enc, gen, dis, X_gt):
    # generator in eval mode
    enc.eval()
    gen.eval()
    dis.eval()

    yz_gt , _, _, _ = enc.forward(X_gt)
    yz_gt = yz_gt.view(X_gt.size()[0], -1)
    rec_categories = yz_gt[:, :nz]

    # loss
    real_recon = gen(rec_categories)
    # feat_real_recon = dis.extract_feat(real_recon)
    # feat_real = dis.extract_feat(X_gt)
    # feat_recon_loss = F.mse_loss(feat_real_recon, feat_real)
    feat_recon_loss = F.mse_loss(255.0 * real_recon, 255.0 * X_gt)
    del real_recon #, feat_real_recon, feat_real
    torch.cuda.empty_cache()

    return rec_categories, feat_recon_loss

root_folder = '/media/vasp/Data1/Users/Peng/intel-image-classification/seg_test/seg_test'
# image_folder = '/media/vasp/Data1/Users/Xiaoming/train-yupenn2'

# category = 'fountain'

Intel6_obj = Intel6(root_folder=root_folder)
image_list = []
label_list = []
correct = []
acc = 0.0

file_image_list = 'image_list_intel.txt'
file_label_list = 'label_list_intel.txt'

image_list = [line.rstrip('\n') for line in open(file_image_list)]
label_list = [line.rstrip('\n') for line in open(file_label_list)]
label_list = list(map(int, label_list))

num_images = len(image_list)

losses = []

post_prob = np.zeros((num_images, num_classes), dtype=np.float32)

for i in range(num_images):
    X_gt = Intel6_obj.load_single_unnormalize(image_list[i])
    X_gt = X_gt.to(DEVICE)


    rec_category, loss = find_single(enc=netE, gen=netG, dis=netD, X_gt=X_gt)

    # loss, rec_category = find_single_8(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)

    rec_category = rec_category.detach().cpu().numpy().squeeze()
    rec_category = rec_category[:num_classes]
    pred_category = np.argmax(rec_category)

    for j in range(num_classes):
        post_prob[i, j] = rec_category[j]

    if pred_category == label_list[i]:
        acc += 1.0
        correct.append(1)
        print('acc=%f'%(acc))
        losses.append(loss.detach().cpu().numpy())
    else:
        correct.append(0)

    # rec_category = rec_category.to(DEVICE)
    # xHAT = netG.forward_1(rec_category)
    # save_image(xHAT.data.cpu(), 'inv.png', normalize=True, nrow=10)

    # if i % 100 == 0:
    #     pickle_name = 'aux_test_all_infer4.pickle'
    #     with open(pickle_name, 'wb') as f:
    #         pickle.dump([post_prob, correct, losses, acc], f)

## last save
correct_Losses = [x / 1.0 for x in losses]
mean = statistics.mean(correct_Losses)
std = statistics.stdev(correct_Losses)
print("Average loss = %.5f, std = %.5f"%(mean, std) )

# print('mean loss=%f'%(sum(losses) / len(losses)))
# pickle_name = 'aux_test_all_infer4.pickle'
# with open(pickle_name, 'wb') as f:
#     pickle.dump([post_prob, correct, losses, acc], f)

Z = np.argmax(post_prob, axis=1)

confuse = np.zeros((num_classes, num_classes)) # confusion matrix
for i in range(num_images):
    confuse[label_list[i], Z[i]] += 1.0

diag_sum = 0
for i in range(num_classes):
    diag_sum += confuse[i, i]

row_sums = np.sum(confuse, axis=1)

## per-category accuracy
for i in range(num_classes):
    for j in range(num_classes):
        confuse[i, j] /= row_sums[i]

np.savetxt("intel_infer_confuse.txt", confuse, fmt="%s")

