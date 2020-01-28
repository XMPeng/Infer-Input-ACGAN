"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
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
from utils import weights_init, compute_acc
from network_Xiaoming import _netG, _netD
# from folder import ImageFolder

from dataload_intel import Intel6


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--load_pretrain', action='store_true', help='load pretrain parameters')
parser.add_argument('--useNoise', action='store_true', help='add a little noise to training data')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

# specify the gpu id if using only 1 gpu
if opt.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id) # 设置使用的GPU，标准操作

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# some hyper parameters
# ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)

# Define the generator and initialize the weights
netG = _netG(nz)
netG.apply(weights_init) # 初始化参数

# Define the discriminator and initialize the weights
netD = _netD(num_classes)
netD.apply(weights_init)

# loss functions
dis_criterion = nn.BCELoss() # 配合Sigmoid()输出使用
aux_criterion = nn.NLLLoss() # 配合多类nn.Softmax()输出

batch_size = opt.batchSize

# tensor placeholders，它们的值会在后面附上
input = torch.FloatTensor(batch_size, 3, 64, 64)
noise = torch.FloatTensor(batch_size, nz)
dis_label = torch.FloatTensor(batch_size)
aux_label = torch.LongTensor(batch_size)

real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise = noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0

epoch = 0

if opt.load_pretrain:
    checkpoint = torch.load(os.path.join(opt.outf, 'checkpoint_file.tar'))
    netG.load_state_dict(checkpoint['G_state_dict'])
    netD.load_state_dict(checkpoint['D_state_dict'])

    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

    epoch = checkpoint['epoch']
    avg_loss_D = checkpoint['avg_loss_D']
    avg_loss_G = checkpoint['avg_loss_G']
    avg_loss_A = checkpoint['avg_loss_A']


root_folder = '/media/vasp/Data1/Users/Peng/intel-image-classification/seg_train/seg_train'
# this_category = 'snowfield'

intel6_obj = Intel6(root_folder=root_folder)
total_batches = intel6_obj.get_total_batches(opt.batchSize)

# for epoch in range(opt.epochs):
while epoch<=opt.epochs:
    intel6_obj.shuffle_data()

    for batch_id in range(total_batches):
        real_cpu, label = intel6_obj.get_next_batch(batch_id, opt.batchSize)
        if opt.useNoise:
            real_cpu = real_cpu+Variable(torch.FloatTensor(real_cpu.size()).normal_(0, 0.01))

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()

        if opt.cuda:
            real_cpu = real_cpu.cuda()

        input.data.copy_(real_cpu) # 将训练真实图像导入input
        dis_label.data.fill_(real_label) # 将dis_label设置为真
        aux_label.data.copy_(label) # 将训练标签导入到aux_label

        dis_output, aux_output = netD(input)

        # 注意张量的维数，dis_label和aux_label的维数都是(batch_size)
        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)

        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean() # 只是对于real/fake的代价

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label) # 这个函数可以用在其他地方，计算准确率

        # train with fake
        # label = np.random.randint(0, num_classes, batch_size)

        # 复用label
        # 以下步骤完成noise的类信息填充过程
        noise_ = np.random.normal(0, 1, (batch_size, nz))

        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1 # 复用label

        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_)) # 转化为张量
        # noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        noise.data.copy_(noise_) # 得到噪声数据

        # 对训练噪声的aux_label进行了赋值
        # aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        ## 复用aux_label

        fake = netG(noise)

        # 对训练噪声的dis_label进行了赋值
        dis_label.data.fill_(fake_label)

        dis_output, aux_output = netD(fake.detach())

        dis_errD_fake = dis_criterion(dis_output, dis_label)
        # aux_errD_fake = aux_criterion(aux_output, aux_label) # 这一项应该是多余的

        # errD_fake = dis_errD_fake + aux_errD_fake

        errD_fake = dis_errD_fake
        errD_fake.backward()

        D_G_z1 = dis_output.data.mean() # 只是对于real/fake的代价

        errD = errD_real + errD_fake # 判别器的总代价
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        dis_label.data.fill_(real_label)  # fake labels are real for generator cost

        ###还是重新生成噪声吧
        noise_ = np.random.normal(0, 1, (batch_size, nz))

        # class_onehot = np.zeros((batch_size, num_classes))
        # class_onehot[np.arange(batch_size), label] = 1  # 复用label

        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))  # 转化为张量
        # noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        noise.data.copy_(noise_)  # 得到噪声数据
        fake = netG(noise)

        dis_output, aux_output = netD(fake)

        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)

        errG = dis_errG + aux_errG # 生成器的总代价
        errG.backward()
        D_G_z2 = dis_output.data.mean() # 只是对于real/fake的代价
        optimizerG.step()

        # compute the average loss
        # curr_iter = epoch * len(dataloader) + i # 这个计算迭代次数的方法有用
        curr_iter = epoch * total_batches + batch_id  # 这个计算迭代次数的方法有用
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter

        all_loss_G += errG.data.item()
        all_loss_D += errD.data.item()
        all_loss_A += accuracy

        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
              % (epoch, opt.epochs, batch_id, total_batches,
                 errD.data.item(), avg_loss_D, errG.data.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
        if curr_iter % 100 == 0:
            # vutils.save_image(
            #     real_cpu, '%s/real_samples.png' % opt.outf)
            # print('Label for eval = {}'.format(eval_label))

            # fake = netG(eval_noise)
            save_image(fake.data, '%s/fake_samples_curr_iter_%03d.png' % (opt.outf, curr_iter),
                       normalize=True)

    # do checkpointing

    torch.save({
        'epoch': epoch,
        'G_state_dict': netG.state_dict(),
        'D_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'avg_loss_D': avg_loss_D,
        'avg_loss_G': avg_loss_G,
        'avg_loss_A': avg_loss_A},
        os.path.join(opt.outf, 'checkpoint_file.tar'))

    epoch = epoch+1


    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
