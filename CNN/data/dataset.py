# coding:utf8
import os
import torch as t
from torch.utils import data
import random

class Apks(data.Dataset):

    def __init__(self, root, train=True, test=False, int_seed=0):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        # 数据集的名字，取后两层目录
        self.name = "_".join(root.split(os.path.sep)[-2:])
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # 按照文件名进行排序
        random.seed(int_seed)
        random.shuffle(imgs)
#        imgs = sorted(imgs)
        # 图片数目
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.8 * imgs_num)]
        else:
            self.imgs = imgs[int(0.8 * imgs_num):]

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = int((img_path.split("_")[-1]).split(".")[-2])
        data = t.load(img_path)
        data = t.unsqueeze(data, dim=0).float()
        return data, label

    def __len__(self):
        return len(self.imgs)