# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import math


class _netG(nn.Module):
    def __init__(self, nz):
        super(_netG, self).__init__()

        self.nz = nz

        self.fc1 = nn.Linear(nz, 512 * 4 * 4)
        self.fc_1 = nn.Linear(100, 8192)
        self.fc_2 = nn.Linear(100, 8192)
        self.fc_3 = nn.Linear(100, 8192)
        self.fc_4 = nn.Linear(100, 8192)
        self.fc_5 = nn.Linear(100, 8192)
        self.fc_6 = nn.Linear(100, 8192)
        self.fc_7 = nn.Linear(100, 8192)
        self.fc_8 = nn.Linear(100, 8192)

        self.prepare_module = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True))

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True), # 输出256x8x8

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),  # 输出128x16x16

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),  # 输出64x32x32

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False), # 输出3x64x64

            nn.Tanh())

    def forward(self, input):
        # 输入batchszxnz
        input = self.fc1(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_1(self, input):
        # 输入batchszxnz
        input = self.fc_1(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_2(self, input):
        # 输入batchszxnz
        input = self.fc_2(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_3(self, input):
        # 输入batchszxnz
        input = self.fc_3(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_4(self, input):
        # 输入batchszxnz
        input = self.fc_4(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_5(self, input):
        # 输入batchszxnz
        input = self.fc_5(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_6(self, input):
        # 输入batchszxnz
        input = self.fc_6(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_7(self, input):
        # 输入batchszxnz
        input = self.fc_7(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output

    def forward_8(self, input):
        # 输入batchszxnz
        input = self.fc_8(input)  # 输出batchszx(512 * 4 * 4)
        input = input.view(-1, 512, 4, 4)
        input = self.prepare_module(input)  # 输出batchszx512x4x4

        output = self.main_module(input)  # 输出batchszx3x64x64
        return output


class _netD(nn.Module):
    def __init__(self, num_classes=3):
        super(_netD, self).__init__()

        self.main_module = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False), #输出64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),  # 输出128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),  # 输出256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),  # 输出512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # discriminator fc
        self.fc_dis = nn.Linear(512*4*4, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(512*4*4, num_classes)

        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.main_module(input)  # 输出batchszx512x4x4
        output = output.view(-1, 512 * 4 * 4)  # 输出batchszx(512*4*4)

        fc_dis = self.fc_dis(output)  # 输出batchszx1
        fc_aux = self.fc_aux(output)  # 输出batchszxnum_classes

        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)


        # fuck = fc_dis.data
        # if torch.sum(torch.isnan(fuck)):
        #     print('??')

        # print('=============================')
        return realfake, classes

    def extract_feature(self, input):
        output = self.main_module(input)  # 输出batchszx512x4x4
        output = output.view(-1, 512 * 4 * 4)  # 输出batchszx(512*4*4)
        return output
