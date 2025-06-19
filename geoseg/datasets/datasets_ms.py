import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import albumentations as albu
from .transform_ms import *


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask, img2, ignore_index):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75,
                                    ignore_index=ignore_index, nopad=False),
                        RandomHorizontalFlip(0.25),
                        RandomVerticalFlip(0.25),
                        ColorJitter(prob=0.25)
                        ])
    img, mask, img2 = crop_aug(img, mask, img2)
    img, mask, img2 = np.array(img), np.array(mask), np.array(img2)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy(), image2=img2.copy())
    img, mask, img2 = aug['image'], aug['mask'], aug['image2']
    return img, mask, img2

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask, img2, ignore_index):
    img, mask, img2 = np.array(img), np.array(mask), np.array(img2)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy(), image2=img2.copy())
    img, mask, img2 = aug['image'], aug['mask'], aug['image2']
    return img, mask, img2

class DatasetMS(Dataset):
    def __init__(self, data_root='/data/test', mode='val', img_dir='image', mask_dir='mask', img2_dir='ir',
                img_suffix='.tif', mask_suffix='.tif', img2_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                img_size=(512, 512), ignore_index=255):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img2_dir = img2_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.img2_suffix = img2_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.ignore_index = ignore_index
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask, img2 = self.load_img_and_mask(index)
            if self.transform:
                img, mask, img2 = self.transform(img, mask, img2, self.ignore_index)
        else:
            img, mask, img2 = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask, img2 = self.transform(img, mask, img2, self.ignore_index)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt=mask, img2=img2)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id[:-4]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img2_name = osp.join(self.data_root, self.img2_dir, img_id + self.img2_suffix)
        img = Image.open(img_name).convert('RGB')
        img2 = Image.open(img2_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return img, mask, img2

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a, img2_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b, img2_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c, img2_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d, img2_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a, img2_a = np.array(img_a), np.array(mask_a), np.array(img2_a)
        img_b, mask_b, img2_b = np.array(img_b), np.array(mask_b), np.array(img2_b)
        img_c, mask_c, img2_c = np.array(img_c), np.array(mask_c), np.array(img2_c)
        img_d, mask_d, img2_d = np.array(img_d), np.array(mask_d), np.array(img2_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = np.floor(w // 2 * (1 - 2 * 0.05))
        strat_y = np.floor(h // 2 * (1 - 2 * 0.05))

        offset_x = random.randint(start_x, np.ceil(w // 2 * (1 + 2 * 0.05)))
        offset_y = random.randint(strat_y,  np.ceil(h // 2 * (1 + 2 * 0.05)))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy(), img2=img2_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy(), img2=img2_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy(), img2=img2_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy(), img2=img2_d.copy())

        img_crop_a, mask_crop_a, img2_crop_a = croped_a['image'], croped_a['mask'], croped_a['image2']
        img_crop_b, mask_crop_b, img2_crop_b = croped_b['image'], croped_b['mask'], croped_b['image2']
        img_crop_c, mask_crop_c, img2_crop_c = croped_c['image'], croped_c['mask'], croped_c['image2']
        img_crop_d, mask_crop_d, img2_crop_d = croped_d['image'], croped_d['mask'], croped_d['image2']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        top_img2 = np.concatenate((img2_crop_a, img2_crop_b), axis=1)
        bottom_img2 = np.concatenate((img2_crop_c, img2_crop_d), axis=1)
        img2 = np.concatenate((top_img2, bottom_img2), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img2 = np.ascontiguousarray(img2)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        img2 = Image.fromarray(img2)

        return img, mask, img2
