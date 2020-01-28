"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
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
from network_Xiaoming import _netG, _netD
# from network_Xiaoming import _netG, _netD
from vgg_16 import VGG16
# from folder import ImageFolder

from random import randrange

# from dataload_Xiaoming_dis import Places9
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
parser.add_argument('--folder_GD', default='./models_intel', help='folder to output images and model checkpoints')
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
netG = _netG(nz)
netD = _netD(num_classes)

checkpoint_GD = torch.load(os.path.join(opt.folder_GD, 'checkpoint_file.tar'))
# checkpoint_GD = torch.load(os.path.join(opt.folder_GD, 'fast.tar'))
netG.load_state_dict(checkpoint_GD['G_state_dict'])
netD.load_state_dict(checkpoint_GD['D_state_dict'])


netG = netG.to(DEVICE)
netD = netD.to(DEVICE)
# netVGG = netVGG.to(DEVICE)

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

def find_single3(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []
    category_probs = torch.zeros((num_test, num_classes), dtype=torch.float32)
    category_probs.requires_grad = False

    rec_categories = torch.zeros((num_test, nz), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, nz).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr

        # Loss = []
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))
            # Loss.append(loss.data.item())

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]
        _, probs = dis.forward(xHAT)
        category_probs[j, :] = probs[0, :]


        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

        # plt.plot(Loss)

    # plt.show()
    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :],

root_folder = '/home/student/Peng/intel-image-classification/seg_test/seg_test'
# image_folder = '/media/vasp/Data1/Users/Xiaoming/train-yupenn2'

# category = 'fountain'

# places9_obj = Places9(root_folder=root_folder)
intel6_obj = Intel6(root_folder=root_folder)
image_list = []
label_list = []
correct = []
acc = 0.0

file_image_list = 'image_list_intel.txt'
file_label_list = 'label_list_intel.txt'

# file_image_list = 'aux_image_list.txt'
# file_label_list = 'aux_label_list.txt'

image_list = [line.rstrip('\n') for line in open(file_image_list)]
label_list = [line.rstrip('\n') for line in open(file_label_list)]
label_list = list(map(int, label_list))

num_images = len(image_list)

losses = []

post_prob = np.zeros((num_images, num_classes), dtype=np.float32)

for i in range(num_images):
    X_gt = intel6_obj.load_single_image(image_list[i])
    X_gt = X_gt.to(DEVICE)


    loss, rec_category = find_single3(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
    # loss, rec_category = find_single_8(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
    losses.append(loss)

    rec_category = rec_category.detach().cpu().numpy().squeeze()
    rec_category = rec_category[:num_classes]
    pred_category = np.argmax(rec_category)

    for j in range(num_classes):
        post_prob[i, j] = rec_category[j]

    if pred_category == label_list[i]:
        acc += 1.0
        correct.append(1)
    else:
        correct.append(0)

    # rec_category = rec_category.to(DEVICE)
    # xHAT = netG.forward_1(rec_category)
    # save_image(xHAT.data.cpu(), 'inv.png', normalize=True, nrow=10)
    if i % 100 == 0:
        pickle_name = 'test_all_intel.pickle'
        with open(pickle_name, 'wb') as f:
            pickle.dump([post_prob, correct, losses, acc], f)

## last save
pickle_name = 'test_all_intel.pickle'
with open(pickle_name, 'wb') as f:
    pickle.dump([post_prob, correct, losses, acc], f)

