import argparse
import os
import numpy as np
import pickle
import time
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from network_Xiaoming import _netG, _netD
# from dataload_Xiaoming_dis import Places9
from dataload_intel import Intel6
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

opt = parser.parse_args()
print(opt)

nz = int(opt.nz)
num_classes = int(opt.num_classes)

## load models
netG = _netG(nz)
netD = _netD(num_classes)

checkpoint_GD = torch.load(os.path.join(opt.folder_GD, 'fast.tar'))
netG.load_state_dict(checkpoint_GD['G_state_dict'])
netD.load_state_dict(checkpoint_GD['D_state_dict'])

netG = netG.to(DEVICE)
netD = netD.to(DEVICE)

def find_single_1(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_1(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt)**2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

def find_single_2(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_2(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

def find_single_3(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_3(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

def find_single_4(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_4(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

def find_single_5(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_5(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

def find_single_6(gen, dis, X_gt, num_test, lr, maxEpochs=100):
    # generator in eval mode
    gen.eval()
    dis.eval()
    category_label = np.zeros(1, dtype=np.int)
    batch_size = 1

    losses = []

    rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
    rec_categories.requires_grad = False

    feat_X_gt = dis.extract_feature(X_gt)
    for j in range(num_test):
        print(j)
        # initialize new noise
        Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
        optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
        start_time = time.time()

        for e in range(maxEpochs):
            # reconstruction loss
            xHAT = gen.forward_6(Zinit)
            feat_xHAT = dis.extract_feature(xHAT)
            # print(Zinit.data)
            recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
            # recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
            # recLoss = F.mse_loss(xHAT, X_gt)
            loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
            print('[%d] loss: %0.5f' % (e, loss.data.item()))

            optZ.zero_grad()
            loss.backward(retain_graph=True)
            optZ.step()

        print('loss: %0.5f' % (loss.data.item()))
        losses.append(loss.data.item())
        rec_categories[j, :] = Zinit[0, :]

        end_time = time.time()
        print('Time elapsed: %0.2f' % (end_time - start_time))

        del Zinit, xHAT, feat_xHAT
        torch.cuda.empty_cache()

    ## best params recovered
    min_ind = losses.index(min(losses))
    return min(losses), rec_categories[min_ind, :]

# def find_single_7(gen, dis, X_gt, num_test, lr, maxEpochs=100):
#     # generator in eval mode
#     gen.eval()
#     dis.eval()
#     category_label = np.zeros(1, dtype=np.int)
#     batch_size = 1
#
#     losses = []
#
#     rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
#     rec_categories.requires_grad = False
#
#     feat_X_gt = dis.extract_feature(X_gt)
#     for j in range(num_test):
#         print(j)
#         # initialize new noise
#         Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
#         optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
#         start_time = time.time()
#
#         for e in range(maxEpochs):
#             # reconstruction loss
#             xHAT = gen.forward_7(Zinit)
#             feat_xHAT = dis.extract_feature(xHAT)
#             # print(Zinit.data)
#             # recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
#             recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
#             # recLoss = F.mse_loss(xHAT, X_gt)
#             loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
#             print('[%d] loss: %0.5f' % (e, loss.data.item()))
#
#             optZ.zero_grad()
#             loss.backward(retain_graph=True)
#             optZ.step()
#
#         print('loss: %0.5f' % (loss.data.item()))
#         losses.append(loss.data.item())
#         rec_categories[j, :] = Zinit[0, :]
#
#         end_time = time.time()
#         print('Time elapsed: %0.2f' % (end_time - start_time))
#
#         del Zinit, xHAT, feat_xHAT
#         torch.cuda.empty_cache()
#
#     ## best params recovered
#     min_ind = losses.index(min(losses))
#     return min(losses), rec_categories[min_ind, :]
#
# def find_single_8(gen, dis, X_gt, num_test, lr, maxEpochs=100):
#     # generator in eval mode
#     gen.eval()
#     dis.eval()
#     category_label = np.zeros(1, dtype=np.int)
#     batch_size = 1
#
#     losses = []
#
#     rec_categories = torch.zeros((num_test, 100), dtype=torch.float32)
#     rec_categories.requires_grad = False
#
#     feat_X_gt = dis.extract_feature(X_gt)
#     for j in range(num_test):
#         print(j)
#         # initialize new noise
#         Zinit = Variable(torch.randn(batch_size, 100).to(DEVICE), requires_grad=True)
#         optZ = torch.optim.RMSprop([Zinit], lr=lr)  # 不同于opt.lr
#         start_time = time.time()
#
#         for e in range(maxEpochs):
#             # reconstruction loss
#             xHAT = gen.forward_8(Zinit)
#             feat_xHAT = dis.extract_feature(xHAT)
#             # print(Zinit.data)
#             # recLoss = F.mse_loss(feat_xHAT, feat_X_gt)
#             recLoss = ((feat_xHAT - feat_X_gt) ** 2).sum()
#             # recLoss = F.mse_loss(xHAT, X_gt)
#             loss = recLoss  # + 200.0*constraintLoss # - (alpha * logProb.mean())
#             print('[%d] loss: %0.5f' % (e, loss.data.item()))
#
#             optZ.zero_grad()
#             loss.backward(retain_graph=True)
#             optZ.step()
#
#         print('loss: %0.5f' % (loss.data.item()))
#         losses.append(loss.data.item())
#         rec_categories[j, :] = Zinit[0, :]
#
#         end_time = time.time()
#         print('Time elapsed: %0.2f' % (end_time - start_time))
#
#         del Zinit, xHAT, feat_xHAT
#         torch.cuda.empty_cache()
#
#     ## best params recovered
#     min_ind = losses.index(min(losses))
#     return min(losses), rec_categories[min_ind, :]


file_image_list = 'image_list_intel.txt'
file_label_list = 'label_list_intel.txt'

image_list = [line.rstrip('\n') for line in open(file_image_list)]
label_list = [line.rstrip('\n') for line in open(file_label_list)]
label_list = list(map(int, label_list))
num_images = len(image_list)

root_folder = '/media/vasp/Data1/Users/Peng/intel-image-classification/seg_test/seg_test'

pickle_name = 'test_all_intel.pickle'
with open(pickle_name, 'rb') as f:
    # [post_prob, correct_old, acc] = pickle.load(f)
    [post_prob, correct_old, _, acc_old] = pickle.load(f)

correct = []
Losses = []
acc = 0.0

intel6_obj = Intel6(root_folder=root_folder)
start_ind = 1500
end_ind = 2800

inds = []
count = 0
predicts = []
for i in range(start_ind, end_ind):
    inds.append(i)
    X_gt = intel6_obj.load_single_image(image_list[i])
    X_gt = X_gt.to(DEVICE)

    zz = np.argsort(post_prob[i, :])
    zz = zz[::-1][:3]
    zz = zz.tolist()

    if not label_list[i] in zz:
        correct.append(0)
        predicts.append(-1)
        Losses.append(10000)
        continue

    losses = []
    for j in range(3):
        if zz[j] == 0:
            loss, rec_category = find_single_1(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        elif zz[j] == 1:
            loss, rec_category = find_single_2(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        elif zz[j] == 2:
            loss, rec_category = find_single_3(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        elif zz[j] == 3:
            loss, rec_category = find_single_4(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        elif zz[j] == 4:
            loss, rec_category = find_single_5(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        elif zz[j] == 5:
            loss, rec_category = find_single_6(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        # elif zz[j] == 6:
        #     loss, rec_category = find_single_7(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)
        # else:
        #     loss, rec_category = find_single_8(gen=netG, dis=netD, X_gt=X_gt, num_test=10, lr=0.01, maxEpochs=5000)

        losses.append(loss)

    min_ind = losses.index(min(losses))
    pred_category = zz[min_ind]
    Losses.append(min(losses))

    predicts.append(pred_category)

    if pred_category == label_list[i]:
        acc += 1.0
        correct.append(1)
    else:
        correct.append(0)

    if count % 50 == 0:
        pickle_name = 'test_specific_intel_1500_2800_.pickle'
        with open(pickle_name, 'wb') as f:
            pickle.dump([inds, predicts, correct, Losses, acc], f)
    count += 1

pickle_name = 'test_specific_intel_1500_2800_.pickle'
with open(pickle_name, 'wb') as f:
    pickle.dump([inds, predicts, correct, Losses, acc], f)






