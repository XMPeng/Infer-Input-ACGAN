# -*- coding: utf-8 -*-
import numpy as np

import os
from os.path import join

from torch.utils import data

from torchvision import transforms, datasets

from PIL import Image
from skimage.io import imread
from skimage import transform
import torch

from operator import itemgetter

import random

from torchvision.utils import save_image

# folders = {'fountain': 0, 'highway': 1, 'iceberg': 2, 'ocean': 3, 'river': 4,
#            'snowfield': 5, 'train_railway': 6, 'volcano': 7, 'wind_farm': 8}

folders = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

# chosen_categories = ('highway', 'train_railway')
# sub_folders = {k: folders[k] for k in chosen_categories}
# sub_folders = {'highway': 0, 'train_railway': 1}

# transform_1 = transforms.Compose(
#     [transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.RandomHorizontalFlip()])
#
# transform_2 = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train_unnormalize = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# transform_train = transforms.Compose(
#     [transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(), \
#      transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class Intel6:
    def __init__(self, root_folder):
        self.image_list = []
        self.label_list = []

        for folder in folders:# sub_folders:
            images = os.listdir(os.path.join(root_folder, folder))
            for image in images:
                if image.endswith('.jpg'):
                    self.image_list.append(os.path.join(root_folder, folder, image))
                    self.label_list.append(folders[folder])
                    # self.label_list.append(sub_folders[folder])

        self.datasz = len(self.label_list)

    def shuffle_data(self):
        rndIdx = np.random.permutation(self.datasz)
        self.image_list = itemgetter(*rndIdx)(self.image_list)
        self.label_list = itemgetter(*rndIdx)(self.label_list)



    def get_next_batch(self, batch_id, batch_sz):
        # 一半数据来自this_category
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 64, 64)
        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.int)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1).squeeze(1)

        # all_imgs = np.zeros((batch_sz, 64, 64,3),dtype=np.uint8)
        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train(im_64)
            X[i, :, :, :] = whole

            # all_imgs[i, :, :, :] = im_64

        return X, y

    def get_next_batch_128(self, batch_id, batch_sz):
        # 一半数据来自this_category
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 128, 128)
        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.int)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1).squeeze(1)

        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_128 = transform.resize(im, (128, 128))
            im_128 = 255 * im_128
            im_128 = im_128.astype(np.uint8)
            whole = transform_train(im_128)
            X[i, :, :, :] = whole

        return X, y

    def get_next_batch2(self, batch_id, batch_sz):
        # 返回归一化和非归一化两种数据
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X_n = torch.empty(batch_sz, 3, 64, 64) # normalized
        X_un = torch.empty(batch_sz, 3, 64, 64) # unnormalized

        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.int)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1).squeeze(1)

        # all_imgs = np.zeros((batch_sz, 64, 64,3),dtype=np.uint8)
        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)

            whole_n = transform_train(im_64)
            X_n[i, :, :, :] = whole_n
            whole_un = transform_train_unnormalize(im_64)
            X_un[i, :, :, :] = whole_un

        return X_n, X_un, y

    def get_next_batch_unnormalize(self, batch_id, batch_sz):
        # 一半数据来自this_category
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 64, 64)
        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.int)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1).squeeze(1)

        # all_imgs = np.zeros((batch_sz, 64, 64,3),dtype=np.uint8)
        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train_unnormalize(im_64)
            X[i, :, :, :] = whole

            # all_imgs[i, :, :, :] = im_64

        return X, y

    def get_next_batch_unnormalize2(self, batch_id, batch_sz):
        # 一半数据来自this_category
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 64, 64)
        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.int)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1).squeeze(1)

        # all_imgs = np.zeros((batch_sz, 64, 64,3),dtype=np.uint8)
        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train_unnormalize(im_64)
            X[i, :, :, :] = whole

            # all_imgs[i, :, :, :] = im_64

        return X, y, image_list_batch, label_list_batch

    def get_next_batch_double(self, batch_id, batch_sz): # 对每个图像都左右翻转
        # 一半数据来自this_category
        image_list_batch = self.image_list[batch_id*batch_sz : (batch_id+1)*batch_sz]
        label_list_batch = self.label_list[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz*2, 3, 64, 64)
        label_list_batch = np.asarray(label_list_batch)
        label_list_batch = label_list_batch.astype(dtype=np.float32)

        y = torch.from_numpy(label_list_batch)
        y = y.view(-1, 1)
        y = torch.stack((y,y),1) # 扩充y
        y = y.view(-1, 1)

        # all_imgs = np.zeros((batch_sz, 64, 64,3),dtype=np.uint8)
        j = 0
        for i in range(batch_sz):
            # print("Process image {}".format(image_list_batch[i]))
            im = imread(image_list_batch[i])

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train(im_64)
            X[j, :, :, :] = whole
            j = j+1

            # 左右翻转
            im = np.fliplr(im)
            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train(im_64)
            X[j, :, :, :] = whole
            j = j + 1

        return X, y



    def get_total_batches(self, batch_sz):
        return self.datasz//batch_sz


    def load_single(self, file_path, category):
        # X = torch.empty(3, 64, 64)
        X = torch.empty(1, 3, 64, 64)
        label = np.zeros(1, dtype=np.int)
        im = imread(file_path)
        im_64 = transform.resize(im, (64, 64))
        im_64 = 255 * im_64
        im_64 = im_64.astype(np.uint8)
        whole = transform_train(im_64)
        X[0, :, :, :] = whole # 3-by-64-by-64

        label[0] = folders[category]
        # label[0] = sub_folders[category]
        y = torch.from_numpy(label)
        # y = y.view(-1, 1).squeeze(1)

        return X, y

    def load_single_image(self, file_path):
        # X = torch.empty(3, 64, 64)
        X = torch.empty(1, 3, 64, 64)
        label = np.zeros(1, dtype=np.int)
        im = imread(file_path)
        im_64 = transform.resize(im, (64, 64))
        im_64 = 255 * im_64
        im_64 = im_64.astype(np.uint8)
        whole = transform_train(im_64)
        X[0, :, :, :] = whole # 3-by-64-by-64

        return X

    def load_single_image_128(self, file_path):
        X = torch.empty(1, 3, 128, 128)
        im = imread(file_path)
        im_128 = transform.resize(im, (128, 128))
        im_128 = 255 * im_128
        im_128 = im_128.astype(np.uint8)
        whole = transform_train(im_128)
        X[0, :, :, :] = whole # 3-by-128-by-128

        return X

    def load_single_unnormalize(self, file_path):
        # X = torch.empty(3, 64, 64)
        X = torch.empty(1, 3, 64, 64)
        label = np.zeros(1, dtype=np.int)
        im = imread(file_path)
        im_64 = transform.resize(im, (64, 64))
        im_64 = 255 * im_64
        im_64 = im_64.astype(np.uint8)
        whole = transform_train_unnormalize(im_64)
        X[0, :, :, :] = whole # 3-by-64-by-64
        return X

    ## 载入某类图像
    def load_category(self, root_folder, category):
        image_list_category = []
        images = os.listdir(os.path.join(root_folder, category))
        for image in images:
            if image.endswith('.jpg'):
                image_list_category.append(os.path.join(root_folder, category, image))

        return image_list_category

        ## 载入某类图像
    # def load_category(self, image_folder, category):
    #     image_list_category = []
    #     images = os.listdir(os.path.join(image_folder, category))
    #     for image in images:
    #         if image.endswith('.jpg'):
    #             image_list_category.append(os.path.join(image_folder, category, image))
    #
    #     return image_list_category

    ## 对某一类图像进行shuffle
    def shuffle_data_category(self, image_list_category):
        datasz = len(image_list_category)
        rndIdx = np.random.permutation(datasz)
        image_list_category = itemgetter(*rndIdx)(image_list_category)

        return image_list_category

    ## 返回某一类图像的batch
    def get_next_batch_category(self, batch_id, batch_sz, image_list_category):
        image_list_batch = image_list_category[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 64, 64)
        for i in range(batch_sz):
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            ## 注意改动
            whole = transform_train(im_64)
            # whole = transform_train_unnormalize(im_64)
            X[i, :, :, :] = whole
        return X

    def get_next_batch_category_un(self, batch_id, batch_sz, image_list_category):
        image_list_batch = image_list_category[batch_id*batch_sz : (batch_id+1)*batch_sz]

        X = torch.empty(batch_sz, 3, 64, 64)
        for i in range(batch_sz):
            im = imread(image_list_batch[i])

            # 控制左右翻转
            if bool(random.getrandbits(1)):
                im = np.fliplr(im)

            im_64 = transform.resize(im, (64, 64))
            im_64 = 255 * im_64
            im_64 = im_64.astype(np.uint8)
            whole = transform_train_unnormalize(im_64)
            X[i, :, :, :] = whole
        return X



