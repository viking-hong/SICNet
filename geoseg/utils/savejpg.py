import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import convolve


def save_first_layer_np(image_np, output_image_path):
    """
    保存四维 NumPy 数组的第一个批次和第一个通道为图像
    :param image_np: 输入的四维 NumPy 数组 (形状为 [batch_size, channels, height, width])
    :param output_image_path: 输出图像路径
    """
    # 检查输入数组的维度
    if image_np.ndim != 4:
        raise ValueError("输入数组必须是四维的，形状为 [batch_size, channels, height, width]")

    # 提取第一个批次和第一个通道的数据
    first_layer = image_np[0, 0, :, :]

    # 归一化到 [0, 255] 范围并转换为 uint8 类型
    first_layer_norm = (first_layer - first_layer.min()) / (first_layer.max() - first_layer.min() + 1e-5) * 255
    first_layer_norm = first_layer_norm.astype(np.uint8)

    # 保存图像
    cv2.imwrite(output_image_path, first_layer_norm)

def save_first_layer(image_tensor, output_image_path):
    first_layer = image_tensor[0, 0, :, :]

    first_layer_np = first_layer.detach().cpu().numpy()
    first_layer_np = (
            (first_layer_np - first_layer_np.min()) / (first_layer_np.max() - first_layer_np.min()) * 255).astype(
        np.uint8)

    cv2.imwrite(output_image_path, first_layer_np)

def save_average_layer_np(image_np, output_image_path):
    """
    保存四维 NumPy 数组的所有通道的平均值为图像
    :param image_np: 输入的四维 NumPy 数组 (形状为 [batch_size, channels, height, width])
    :param output_image_path: 输出图像路径
    """
    # 检查输入数组的维度
    if image_np.ndim != 4:
        raise ValueError("输入数组必须是四维的，形状为 [batch_size, channels, height, width]")

    # 提取第一个批次，并对所有通道取平均值
    average_layer = image_np[0].mean(axis=0)

    # 归一化到 [0, 255] 范围并转换为 uint8 类型
    average_layer_norm = (average_layer - average_layer.min()) / (average_layer.max() - average_layer.min() + 1e-5) * 255
    average_layer_norm = average_layer_norm.astype(np.uint8)

    # 保存图像
    cv2.imwrite(output_image_path, average_layer_norm)

def save_average_layer(image_tensor, output_image_path):
    # 计算所有层的平均值
    average_layer = image_tensor.mean(dim=1)[0]  # 对第2维 (通道) 求平均

    # 转换为 numpy 数组
    average_layer_np = average_layer.detach().cpu().numpy()

    # 归一化并转换为 8-bit 图像
    average_layer_np = (
        (average_layer_np - average_layer_np.min()) /
        (average_layer_np.max() - average_layer_np.min()) * 255
    ).astype(np.uint8)

    # 保存图像
    cv2.imwrite(output_image_path, average_layer_np)

def save_single_band_heatmap(mask_tensor, output_image_path):
    first_layer = mask_tensor[0, 0, :, :]
    mask_tensor = first_layer.float().unsqueeze(0).unsqueeze(0)

    kernel = torch.ones(1, 1, 12, 12).float() / 144  # 3x3均值滤波核
    smoothed_image = F.conv2d(mask_tensor, kernel.to('cuda'), padding=0)
    mask_tensor_pooled = smoothed_image.squeeze(0).squeeze(0)

    # 将张量转换为 NumPy 数组并归一化到 [0, 1]
    mask_np = mask_tensor_pooled.detach().cpu().numpy()
    mask_np = cv2.GaussianBlur(mask_np, (5, 5), sigmaX=1)
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-5)  # 防止除零错误

    # 使用渐变色 cmap 生成热点图

    plt.imshow(mask_np, cmap='coolwarm')  # 使用 'coolwarm' 或其他渐变色
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()

def save_single_band_heatmap_np(mask_np, output_image_path):
    """
    对四维 NumPy 数组的第一个批次和第一个通道生成热点图并保存
    :param mask_np: 输入的四维 NumPy 数组 (形状为 [batch_size, channels, height, width])
    :param output_image_path: 输出图像路径
    """
    # 检查输入数组的维度
    if mask_np.ndim != 4:
        raise ValueError("输入数组必须是四维的，形状为 [batch_size, channels, height, width]")

    # 提取第一个批次和第一个通道的数据
    first_layer = mask_np[0, 0, :, :]

    # 均值滤波核
    kernel = np.ones((12, 12)) / 144  # 12x12 均值滤波核

    # 使用卷积进行均值滤波
    smoothed_image = convolve(first_layer, kernel, mode='constant')

    # 高斯模糊
    smoothed_image = cv2.GaussianBlur(smoothed_image, (5, 5), sigmaX=1)

    # 归一化到 [0, 1]
    smoothed_image = (smoothed_image - smoothed_image.min()) / (smoothed_image.max() - smoothed_image.min() + 1e-5)  # 防止除零错误

    # 使用渐变色 cmap 生成热点图
    plt.imshow(smoothed_image, cmap='coolwarm')  # 使用 'coolwarm' 或其他渐变色
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()

def save_average_band_heatmap(mask_tensor, output_image_path):
    # 对所有通道求平均
    avg_layer = mask_tensor.mean(dim=1, keepdim=True)  # 在通道维度上求均值

    # 平滑处理 - 使用 12x12 平均滤波核
    kernel = torch.ones(1, 1, 12, 12).float().to(mask_tensor.device) / 144
    smoothed_image = F.conv2d(avg_layer.float(), kernel, padding=0)

    # 转换为 NumPy 数组并归一化到 [0, 1]
    mask_np = smoothed_image.squeeze(0).squeeze(0).detach().cpu().numpy()
    mask_np = cv2.GaussianBlur(mask_np, (5, 5), sigmaX=1)
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-5)

    # ones_matrix = np.ones_like(mask_np)
    # mask_np = ones_matrix - mask_np

    # 生成热图
    plt.imshow(mask_np, cmap='coolwarm')  # 使用 'coolwarm' 或其他渐变色
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_average_band_heatmap_np(mask_np, output_image_path):
    """
    对四维 NumPy 数组的所有通道求平均值并生成热点图
    :param mask_np: 输入的四维 NumPy 数组 (形状为 [batch_size, channels, height, width])
    :param output_image_path: 输出图像路径
    """
    # 检查输入数组的维度
    if mask_np.ndim != 4:
        raise ValueError("输入数组必须是四维的，形状为 [batch_size, channels, height, width]")

    # 对所有通道求平均
    avg_layer = mask_np[0].mean(axis=0)

    # 均值滤波核
    kernel = np.ones((12, 12)) / 144  # 12x12 均值滤波核

    # 使用卷积进行均值滤波
    smoothed_image = convolve(avg_layer, kernel, mode='constant')

    # 高斯模糊
    smoothed_image = cv2.GaussianBlur(smoothed_image, (5, 5), sigmaX=1)

    # 归一化到 [0, 1]
    smoothed_image = (smoothed_image - smoothed_image.min()) / (smoothed_image.max() - smoothed_image.min() + 1e-5)  # 防止除零错误

    # ones_matrix = np.ones_like(smoothed_image)
    # smoothed_image = ones_matrix - smoothed_image

    # 使用渐变色 cmap 生成热点图
    plt.imshow(smoothed_image, cmap='coolwarm')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()