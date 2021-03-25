# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import init_weights
from models.layers import conv2d_down_block, conv2d_up_block


class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv2d_down_block(self.in_channels, filters[0], self.is_batchnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv2d_down_block(filters[0], filters[1], self.is_batchnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv2d_down_block(filters[1], filters[2], self.is_batchnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv2d_down_block(filters[2], filters[3], self.is_batchnorm)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.center = conv2d_down_block(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = conv2d_up_block(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = conv2d_up_block(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = conv2d_up_block(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = conv2d_up_block(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.pool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.pool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.pool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.pool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)
