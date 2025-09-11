# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_mean, style_std, alpha=0.6):  # 读取一个风格
        size = content_feat.size()
        content_mean, content_std = content_feat.mean([2, 3]), content_feat.std([2, 3])
        normalized_feat = (content_feat - content_mean.view(size[0], size[1], 1, 1)) / (
                content_std.view(size[0], size[1], 1, 1) + 1e-8)
        style_feat = normalized_feat * style_std + style_mean
        # 使用 alpha 控制内容特征与风格特征之间的混合比例
        output = alpha * style_feat + (1 - alpha) * content_feat

        return output

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        # Unet
        basic_layer = args.n_basic_layer
        self.conv0 = EncoderConv(args.n_channels, basic_layer)
        self.conv1 = EncoderConv(basic_layer, basic_layer)

        self.conv2 = EncoderConv(basic_layer, 2*basic_layer)
        self.conv3 = EncoderConv(2*basic_layer, 2*basic_layer)

        self.conv4 = EncoderConv(2*basic_layer, 4*basic_layer)
        self.conv5 = EncoderConv(4*basic_layer, 4*basic_layer)

        self.conv6 = EncoderConv(4*basic_layer, 8*basic_layer)
        self.conv7 = EncoderConv(8*basic_layer, 8*basic_layer)

        self.conv8 = DecoderConv(12*basic_layer, 4*basic_layer)
        self.conv9 = DecoderConv(4*basic_layer, 4*basic_layer)

        self.conv10 = DecoderConv(6*basic_layer, 2*basic_layer)
        self.conv11 = DecoderConv(2*basic_layer, 2*basic_layer)

        self.conv12 = DecoderConv(3*basic_layer, basic_layer)
        self.conv13 = DecoderConv(basic_layer, basic_layer)

        self.out_conv = nn.Conv2d(basic_layer, args.n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

        self.adain = AdaIN()# AdaIN for style transfer

    def forward(self, x, style_mean=None, style_std=None):
        # block0
        x_0_0 = self.conv0(x)
        x_0_1 = self.conv1(x_0_0)
        # print(x_0_1.flatten())

        # block1
        x = self.maxpooling(x_0_1)
        x_1_0 = self.conv2(x)
        x_1_1 = self.conv3(x_1_0)

        # block2
        x = self.maxpooling(x_1_1)
        x_2_0 = self.conv4(x)
        x_2_1 = self.conv5(x_2_0)

        # block3
        x = self.maxpooling(x_2_1)
        x_3_0 = self.conv6(x)
        x_3_1 = self.conv7(x_3_0)

        # Apply AdaIN if style_mean and style_std are provided
        if style_mean is not None and style_std is not None:
            x_3_1 = self.adain(x_3_1, style_mean, style_std)

        # block4
        x = self.up(x_3_1)
        x_2_2 = self.conv8(cat([x, x_2_1], dim=1))
        x_2_3 = self.conv9(x_2_2)

        # block5
        x = self.up(x_2_3)
        x_1_2 = self.conv10(cat([x, x_1_1], dim=1))
        x_1_3 = self.conv11(x_1_2)

        # block6
        x = self.up(x_1_3)
        x_0_2 = self.conv12(cat([x, x_0_1], dim=1))
        x_0_3 = self.conv13(x_0_2)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out

    def get_first_layer_output(self, x):
        x = self.conv0(x)
        x = nn.ReLU()(x)
        return x

    def get_deep_layer_output(self, x):
        # block0
        x_0_0 = self.conv0(x)
        x_0_1 = self.conv1(x_0_0)

        # block1
        x = self.maxpooling(x_0_1)
        x_1_0 = self.conv2(x)
        x_1_1 = self.conv3(x_1_0)

        # block2
        x = self.maxpooling(x_1_1)
        x_2_0 = self.conv4(x)
        x_2_1 = self.conv5(x_2_0)

        # block3
        x = self.maxpooling(x_2_1)
        x_3_0 = self.conv6(x)
        x_3_1 = self.conv7(x_3_0)

        return x_3_1