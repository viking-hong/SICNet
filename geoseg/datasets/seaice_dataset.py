import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ('No Change', 'Change')
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def aug(image1, image2, mask):
    image1, image2, mask = np.array(image1), np.array(image2), np.array(mask)
    return image1, image2, mask


def swap_region(img1, img2, mask=None, threshold=0.2):
    height, width = img1.shape[:2]

    # 随机生成区域大小
    region_width = random.randint(width // 10, width // 2)
    region_height = random.randint(height // 10, height // 2)

    # 初始化最佳区域的变量
    best_weight = 0.5
    best_x, best_y = 0, 0

    # 遍历所有可能的区域来寻找权重最大的区域
    for _ in range(10):  # 迭代次数，可以根据需求调整
        x = random.randint(0, width - region_width)
        y = random.randint(0, height - region_height)

        # 计算区域内的显著性权重和
        region_weight = np.sum(mask[y:y + region_height, x:x + region_width])#区域内变化总数

        region_weight = abs((region_width * region_height) / 2- region_weight) / (region_width * region_height)#距离0.5差多少

        # 更新最佳权重区域
        if region_weight < best_weight:
            best_weight = region_weight
            best_x, best_y = x, y

    # 检查显著区域权重是否足够显著，否则回退到随机选择
    if best_weight < threshold:
        # 随机选择区域
        best_x = random.randint(0, width - region_width)
        best_y = random.randint(0, height - region_height)

    # 提取并交换两幅图像中选定区域
    region1 = img1[best_y:best_y + region_height, best_x:best_x + region_width].copy()
    region2 = img2[best_y:best_y + region_height, best_x:best_x + region_width].copy()

    img1[best_y:best_y + region_height, best_x:best_x + region_width] = region2
    img2[best_y:best_y + region_height, best_x:best_x + region_width] = region1

    return img1, img2


class SeaIceDataset(Dataset):
    def __init__(self, data_root='data/vaihingen/test', mode='val', image1_dir='date1', image2_dir='date2', mask_dir='label',
                 img_suffix='.jpg', mask_suffix='.jpg', mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE, mixup=False):
        self.data_root = data_root
        self.image1_dir = image1_dir
        self.image2_dir = image2_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.image1_dir, self.image2_dir, self.mask_dir)
        self.mixup = mixup

    def __getitem__(self, index):
        image1, image2, mask = self.load_img_and_mask(index)
        image1, image2, mask = aug(image1, image2, mask)
        if self.mixup:
            image1, image2 = swap_region(image1, image2, mask)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, image1=image1, image2=image2, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, image1_dir, image2_dir, mask_dir):
        image1_filename_list = os.listdir(osp.join(data_root, image1_dir))
        image2_filename_list = os.listdir(osp.join(data_root, image2_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(image1_filename_list) == len(image2_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        image1_name = osp.join(self.data_root, self.image1_dir, img_id + self.img_suffix)
        image2_name = osp.join(self.data_root, self.image2_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        image1 = Image.open(image1_name).convert('RGB')
        image2 = Image.open(image2_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        mask = np.array(mask) / 255
        mask = mask.astype(np.uint8)
        return image1, image2, mask


