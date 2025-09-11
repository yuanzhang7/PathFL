# -*- coding: utf-8 -*-
from os.path import exists, join
from os import listdir
import torch
from torch.utils import data
import numpy as np
from Transform import transform_img_lab
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])


def read_file_from_txt(txt_path, base_dir=None):
    if base_dir is not None and not os.path.isabs(txt_path):
        txt_path = os.path.join(base_dir, txt_path)
    files = []
    for line in open(txt_path, 'r'):
        files.append(line.strip())
    return files


class Dataloader(data.Dataset):
    def __init__(self, args, center_id=None):
        super(Dataloader, self).__init__()
        # 根据center_id或args动态选择数据路径
        if center_id is not None:
            image_txt = getattr(args, f'c{center_id}_train_image_txt')
            label_txt = getattr(args, f'c{center_id}_train_label_txt')
        else:
            image_txt = args.c_train_image_txt
            label_txt = args.c_train_label_txt
        self.image_file = read_file_from_txt(image_txt, getattr(args, 'base_dir', None))
        self.label_file = read_file_from_txt(label_txt, getattr(args, 'base_dir', None))
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]
        
        # 确保路径正确拼接base_dir
        if getattr(self.args, 'base_dir', None) is not None:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.args.base_dir, image_path)
            if not os.path.isabs(label_path):
                label_path = os.path.join(self.args.base_dir, label_path)
        
        name = image_path[image_path.rfind('/') + 1:]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        image = image.transpose(2, 0, 1)
        label = Image.open(label_path).convert('L')
        label = np.array(label, dtype=np.float32)

        if label.shape[0]==3:
            label=label[0]#3 channel


        _, y, x = image.shape
        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)
        label = np.where(label > 0, 1, 0)   #label [0,1] binary
        # 确保标签是int类型用于二值分割
        label = label.astype(dtype=np.int32)

        mean, std = np.load(self.args.Meanstd_dir)
        mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        # print('mean.shape', mean.shape, 'mean', mean)

        image = (image-mean)/std
        # label = label / (np.max(label))
        # image_crop from left top
        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int32)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int32)[0]
        image = image[:, center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]
        label = label[center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]

        # image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict['image']
        label_trans = data_dict['label']
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()
        
        # # # 确保数据增强后标签仍然是int类型
        # if label_trans.dtype != np.int32:
        #     label_trans = label_trans.astype(np.int32)
        if label_trans.dtype != np.float32:
            label_trans = label_trans.astype(np.float32)

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)