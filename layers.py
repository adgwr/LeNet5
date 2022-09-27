import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import unfold
from sklearn.metrics import accuracy_score
import copy
import math



# 自定义卷积层
class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1):
        super(MyConv, self).__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__dilation = dilation
        # 定义参数
        self.__conv = nn.Parameter(torch.Tensor(self.__kernel_size, self.__kernel_size))
        self.__bias = nn.Parameter(torch.Tensor(self.__out_channels))
        # 参数初始化
        nn.init.kaiming_uniform_(self.__conv, a=math.sqrt(5))
        nn.init.uniform_(self.__bias)

    def forward(self, input):
        _, _, H, W = input.shape
        # 对input进行im2col操作
        input = unfold(input, kernel_size=self.__kernel_size, stride=self.__stride,
                         padding=self.__padding, dilation=self.__dilation)
        # 对卷积核进行im2col操作
        conv_unfold = self.__conv.view(self.__kernel_size*self.__kernel_size)
        conv_unfold = torch.cat((conv_unfold,) * self.__in_channels, dim=0)
        conv_unfold = torch.stack((conv_unfold,) * self.__out_channels, dim=0)
        # 利用矩阵乘法完成卷积
        input = torch.matmul(conv_unfold, input)
        # 将结果恢复为原来的维度
        out_H = int((H + 2 * self.__padding - self.__kernel_size) / self.__stride + 1)
        out_W = int((W + 2 * self.__padding - self.__kernel_size) / self.__stride + 1)
        input = input.view(-1, self.__out_channels, out_H, out_W)
        # 添加偏置项
        input = input + self.__bias.view(self.__out_channels, 1, 1)
        return input


# 自定义平均池化层
class MyAvgPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MyAvgPool, self).__init__()
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

    def forward(self, input):
        N, C, H, W = input.shape
        # 对input进行im2col操作, 展开为(N, C * kernel_H * kernel_W, 滑动次数)
        input = unfold(input, kernel_size=self.__kernel_size, stride=self.__stride,
                         padding=self.__padding)
        # 构造平均池化的kernel展开的tensor
        avg_kernel_area = self.__kernel_size * self.__kernel_size
        avg_fractor = 1 / avg_kernel_area
        avg_tensor = torch.zeros(C, input.shape[1])
        for i in range(len(avg_tensor)):
            avg_tensor[i, i * avg_kernel_area: (i + 1) * avg_kernel_area] = avg_fractor
        # 利用矩阵乘法完成平均池化
        input = torch.matmul(avg_tensor.cuda(), input)
        # 将结果恢复为原来的维度
        input = input.view(N, C, H // self.__kernel_size, W // self.__kernel_size)
        return input


# 自定义线性层
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.__in_features = in_features
        self.__out_features = out_features
        # 定义参数
        self.__weight = nn.Parameter(torch.Tensor(self.__out_features, self.__in_features))
        if bias:
            self.__bias = nn.Parameter(torch.Tensor(self.__out_features))
            nn.init.uniform_(self.__bias)
        # 初始化参数
        nn.init.kaiming_uniform_(self.__weight, a=math.sqrt(5))

    def forward(self, input):
        res = torch.matmul(input, self.__weight.t())
        if self.__bias is not None:
            res += self.__bias
        return res




