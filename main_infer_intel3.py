import argparse
# from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model_64_infer3 import *
import os
import numpy as np
import torch.nn.functional as F

from dataload_intel import Intel6

batch_size = 128
lr = 0.0002
latent_size = 106
num_classes = 6
num_epochs = 300
cuda_device = "0"

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=boolean_string, default=True) # 有gpu改为True
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

def compute_acc(preds, labels):
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct)/ float(len(labels.data)) * 100.0
    return acc


netE = tocuda(Encoder(latent_size, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, num_classes, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

root_folder = '/media/vasp/Data1/Users/Peng/intel-image-classification/seg_train/seg_train'

intel6_obj = Intel6(root_folder=root_folder)
total_batches = intel6_obj.get_total_batches(batch_size)

Iter = 0

aux_label = torch.LongTensor(batch_size)
aux_label = tocuda(aux_label)
aux_label = Variable(aux_label)

z_fake = torch.FloatTensor(batch_size, latent_size)
z_fake = tocuda(z_fake)
z_fake = Variable(z_fake)

for epoch in range(num_epochs):
    intel6_obj.shuffle_data()
    i = 0
    for batch_id in range(total_batches): # for (data, target) in train_loader:
        real_cpu, label = intel6_obj.get_next_batch_unnormalize(batch_id, batch_size)
        real_cpu = tocuda(real_cpu)
        aux_label.data.copy_(label)

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(real_cpu.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(real_cpu.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(real_cpu))

        if real_cpu.size()[0] != batch_size:
            continue

        d_real = Variable(real_cpu)

        noise_ = np.random.normal(0, 1, (batch_size, latent_size))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1  # 复用label
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))  # 转化为张量
        z_fake.data.copy_(noise_)

        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        output_real, class_res_res = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, class_res_fake = netD(d_fake + noise2, z_fake.view(batch_size, latent_size, 1, 1))

        # output_real, class_res_res = netD(d_real + noise1, output_z)
        # output_fake, class_res_fake = netD(d_fake + noise2, z_fake)
        real_recon = netG(mu)
        # feat_real_recon = netD.extract_feat(real_recon)
        # feat_real = netD.extract_feat(d_real)
        # feat_recon_loss = F.mse_loss(feat_real_recon, feat_real) / feat_real_recon.numel()
        feat_recon_loss = F.mse_loss(real_recon, d_real) #/ d_real.numel()

        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label) + aux_criterion(class_res_res, aux_label) + 10*feat_recon_loss
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label) + aux_criterion(class_res_fake, aux_label)
        acc = compute_acc(class_res_res, aux_label)

        if loss_g < 3.5:# loss_g.data[0] < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        # if i % 1 == 0:
        #     print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.data[0], "G loss :", loss_g.data[0],
        #           "D(x) :", output_real.mean().data[0], "D(G(x)) :", output_fake.mean().data[0])

        if i % 1 == 0:
            # print("Epoch :", epoch, "Iter :", i, "D Loss %.4f:", loss_d.data.item(), "G loss %.4f:", loss_g.data.item(),
            #       "D(x) %.4f:", output_real.mean().data.item(), "D(G(x)) %.4f:", output_fake.mean().data.item())

            print(
                'Epoch: %d  Iter: %d D Loss: %.4f G loss: %.4f D(x): %.4f D(G(z)): %.4f acc: %.4f reconloss: %.4f'
                % (epoch, i, loss_d.data.item(), loss_g.data.item(), output_real.mean().data.item(),
                   output_fake.mean().data.item(), acc, feat_recon_loss.data.item())
            )

        # if i % 50 == 0:
        #     vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
        #     vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (opt.save_image_dir))

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu(), './%s/fake_%d.png' % (opt.save_image_dir, Iter))
            # vutils.save_image(d_real.cpu(), './%s/real_%d.png'% (opt.save_image_dir, Iter))

        i += 1

        Iter += 1

    if epoch % 1 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch.pth' % (opt.save_model_dir))
        torch.save(netE.state_dict(), './%s/netE_epoch.pth' % (opt.save_model_dir))
        torch.save(netD.state_dict(), './%s/netD_epoch.pth' % (opt.save_model_dir))

        # vutils.save_image(d_fake.cpu(), './%s/fake_%d.png' % (opt.save_image_dir, epoch))
