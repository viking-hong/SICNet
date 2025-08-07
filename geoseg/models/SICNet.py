# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

from torch.nn.modules.padding import ReplicationPad2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import math
from geoseg.utils.Conv import *
import cv2
from geoseg.utils.Conv import *
from geoseg.utils.savejpg import *
import matplotlib.pyplot as plt


class CrossAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )


        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='constant')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='constant')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant')  # reflect
        return x

    def forward(self, x1, x2):
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        x1 = x1.to('cuda')
        x2 = x2.to('cuda')
        B, C, H, W = x1.shape

        x1 = self.pad(x1, self.ws)
        B, C, Hp, Wp = x1.shape
        qkv1 = self.qkv(x1)

        x2 = self.pad(x2, self.ws)
        qkv2 = self.qkv(x2)

        q1, _, v1 = rearrange(qkv1, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        _, k1, v2 = rearrange(qkv2, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                              d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws,
                              ws2=self.ws)

        dots = (q1 @ k1.transpose(-2,
                                -1)) * self.scale  #######torch.Size([64, 8]) torch.Size([8, 64]) -> torch.Size([64, 64])

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # print(relative_position_bias.unsqueeze(0).shape)
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        ones = torch.ones_like(attn)
        attn = (attn @ v1 + (ones - attn) @ v2) / 2

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out.detach().cpu().numpy()

class laplace(nn.Module):
    def __init__(self, alpha=0.5, inchannels=64):
        super(laplace, self).__init__()
        self.alpha = alpha
        self.conv = ConvBNRepVGG(in_channels=inchannels, out_channels=inchannels, kernel_size=3)

    def laplacian_operator(self, tensor):
        # 定义拉普拉斯核
        laplacian_kernel = torch.tensor(
            # [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
            dtype=tensor.dtype, device=tensor.device
        ).unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, 3, 3]

        # 扩展拉普拉斯核以适配输入通道
        laplacian_kernel = laplacian_kernel.repeat(tensor.shape[1], 1, 1, 1)  # [groups, 1, 3, 3]

        # 执行分组卷积
        return F.conv2d(tensor, laplacian_kernel, padding=1, groups=tensor.shape[1])

    def gaussian_kernel(self, size: int, sigma: float):
        """生成高斯滤波器"""
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    def gaussian_filter(self, tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0):
        kernel = self.gaussian_kernel(kernel_size, sigma)

        # 扩展高斯核以匹配输入张量的通道数
        in_channels = tensor.shape[1]  # 获取输入张量的通道数
        kernel = kernel.repeat(in_channels, 1, 1, 1)
        kernel = kernel.to('cuda')

        filtered_tensor = F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=in_channels)

        return filtered_tensor

    def forward(self, tensor):
        tensor = torch.from_numpy(tensor)
        tensor = tensor.to('cuda')
        # Compute Laplacian high-frequency components
        laplace = self.gaussian_filter(tensor)
        laplace = self.laplacian_operator(laplace)
        fused_high_freq = self.conv(laplace)
        return fused_high_freq.detach().cpu().numpy()

import pywt
import pytorch_wavelets as pw
class wcaf(nn.Module):
    def __init__(self, inchannels=256):
        super(wcaf, self).__init__()
        self.ca = CrossAttention(dim=inchannels)
        self.log = laplace(inchannels=inchannels)
        self.repvgg = ConvBNRepVGGReLU(in_channels=inchannels, out_channels=inchannels)

    def forward(self, input1, input2):
        input1 = self.repvgg(input1)
        input2 = self.repvgg(input2)
        save_average_band_heatmap(abs(input1 - input2), 'D:/Dataset-view/SeaIceCD_bh/view/out1.jpg')
        save_average_band_heatmap(input1, 'D:/Dataset-view/SeaIceCD_bh/view/input1.jpg')
        save_average_band_heatmap(input2, 'D:/Dataset-view/SeaIceCD_bh/view/input2.jpg')
        [ca1, (ch1, cv1, cd1)] = pywt.dwt2(input1.detach().cpu().numpy(), 'haar')
        [ca2, (ch2, cv2, cd2)] = pywt.dwt2(input2.detach().cpu().numpy(), 'haar')
        save_average_band_heatmap_np(ca1, 'D:/Dataset-view/SeaIceCD_bh/view/ca1.jpg')
        save_average_band_heatmap_np(ca2, 'D:/Dataset-view/SeaIceCD_bh/view/ca2.jpg')

        ca = self.ca(ca1, ca2)
        save_average_band_heatmap_np(ca, 'D:/Dataset-view/SeaIceCD_bh/view/ca.jpg')

        ch1 = self.log(ch1)
        cv1 = self.log(cv1)
        cd1 = self.log(cd1)

        ch2 = self.log(ch2)
        cv2 = self.log(cv2)
        cd2 = self.log(cd2)

        input1 = pywt.idwt2((ca, (ch1, cv1, cd1)), 'haar')
        input2 = pywt.idwt2((ca, (ch2, cv2, cd2)), 'haar')
        input1 = torch.from_numpy(input1)
        input1 = input1.to('cuda')
        input2 = torch.from_numpy(input2)
        input2 = input2.to('cuda')
        save_average_band_heatmap(input1, 'D:/Dataset-view/SeaIceCD_bh/view/input12.jpg')
        save_average_band_heatmap(input2, 'D:/Dataset-view/SeaIceCD_bh/view/input22.jpg')

        input1 = self.repvgg(input1)
        input2 = self.repvgg(input2)

        tensor = abs(input1 - input2)
        save_average_band_heatmap(tensor, 'D:/Dataset-view/SeaIceCD_bh/view/out2.jpg')
        tensor = self.repvgg(tensor)
        return tensor


class SICNet(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SICNet, self).__init__()

        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

        self.laplace1 = wcaf(inchannels=16)
        self.laplace2 = wcaf(inchannels=32)
        self.laplace3 = wcaf(inchannels=64)
        self.laplace4 = wcaf(inchannels=128)

    def forward(self, x1, x2):
        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        # x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x4d = torch.cat((pad4(x4d), self.laplace4(x43_1, x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        # x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x3d = torch.cat((pad3(x3d), self.laplace3(x33_1, x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        # x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x2d = torch.cat((pad2(x2d), self.laplace2(x22_1, x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        # x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x1d = torch.cat((pad1(x1d), self.laplace1(x12_1, x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        #out = self.sm(x11d)

        output = []
        output.append(x11d)

        return output[0]
