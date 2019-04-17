# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:41:49 2019

@author: ZhangXin
"""
import os
import time

import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torchnet import meter
import random


# dataset
class AndroidDataset(Dataset):
    def __init__(self, root="", train=True, test=False, data_pickle=""):
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # imgs = sorted(imgs)
        random.shuffle(imgs)
        
        imgsNum = len(imgs)
        
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgsNum)]
        else:
            self.imgs = imgs[int(0.7 * imgsNum):]
    
    
    def _genPictureData(featureListFilePath):
        pass
    
    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = int(imgPath.split("_")[-1].split(".")[-2])
        data = torch.load(imgPath)
        data = torch.unsqueeze(data, dim=0).float()
        return data, label
    
    def __len__(self):
        return len(self.imgs)



# model
class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(                         # (1, 300, 4)
            nn.Conv2d(1, 64, kernel_size=4),                   # (64, 297, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (64, 99, 1)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),      # (192, 99, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (192, 33, 1)
            nn.Conv2d(192, 384, kernel_size=5, padding=2),     # (384, 33, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=5, padding=2),     # (256, 33, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),     # (256, 33, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (256, 11, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def save_model(model, filename="model.ckpt"):
    checkpoint = {"state_dict": model.state_dict()}
    with open(filename, "wb") as f:
        torch.save(checkpoint, f)

def load_model(filename):
    with open(filename, "rb") as f:
        checkpoint = torch.load(f)
    model = AlexNet()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def validation_accuracy(model, dataloader):
    # model.eval()
    correct_num, sum_num = 0, 0
    for i, (val_input, label) in enumerate(dataloader):
        if torch.cuda.is_available():
            val_input = val_input.cuda()
            label = label.cuda()
#        print("val_input:")
#        print(val_input)
#        print("val_input size:")
#        print(val_input.size())
#        print("label:")
#        print(label)
#        print("label size:")
#        print(label.size())        
        output = model(val_input)
#        print("output:")
#        print(output)
#        print("output size:")
#        print(output.size())

        pred_result = torch.max(output, 1)[1].detach().squeeze().cpu().numpy()
#        print("pred_result:")
#        print(pred_result)

        correct_num += (pred_result == label.detach().cpu().numpy()).astype(int).sum()
#        print("correct num:")
#        print((pred_result == label.detach().cpu().numpy()).astype(int).sum())
#        print("label.size(0):")
#        print(label.size(0))
        sum_num += label.size(0)

    # model.train()

    accuracy = float(correct_num) / float(sum_num)
    print("correct: %s, sum: %s" % (correct_num, sum_num))
    
    return accuracy



# train
def train(root, data_pickle, n_epochs, print_every, lr, batch_size, load_model_path):
    train_dataset = AndroidDataset(root=root)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    
    validation_dataset = AndroidDataset(root=root, train=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers = 4)
    
    
    if os.path.isfile(load_model_path):
        model = load_model(load_model_path)
    else:
        model = AlexNet()
    
    print(model)
#    if torch.cuda.device_count() > 1:
#        # print("Let's use", torch.cuda.device_count(), "GPUs!")
#        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    print(optimizer)
    print(criterion)

    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    
    try:
        print("Training for %d epochs..." % n_epochs)
        for epoch in range(n_epochs):
            
            loss_meter.reset()
            
            for i, (data, label) in enumerate(train_dataloader):
                label = label.long()
                input = Variable(data)
                target = Variable(label)
                
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
                
                optimizer.zero_grad()
                output= model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                loss_meter.add(loss.item())
                
                if (i + 1) % print_every == 0:
                    print("Epoch: %s | percentage : %.2f%% | train loss: %.4f"
                          % (epoch, (100. * i / len(train_dataloader)), loss_meter.value()[0]))
            
            val_accuracy = validation_accuracy(model, validation_dataloader)
            print("Epoch: %s | lr: %s |train loss: %.4f| validation_accuracy: %.4f"
                  % (epoch, lr, loss_meter.value()[0], val_accuracy))
#            print("Epoch: %s | lr: %s |train loss: %.4f"
#                  % (epoch, lr, loss_meter.value()[0]))   

            save_model(model, "%sepoch_%s_loss_%.4f_accuracy_%.4f_time_%s.ckpt" % 
                       ("./checkpoints/", epoch, loss_meter.value()[0], val_accuracy,
                        time.strftime("%Y%m%d%H%M", time.localtime())))
            
            if loss_meter.value()[0] > previous_loss:
                lr = lr * 0.5
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            
            previous_loss = loss_meter.value()[0]
    except KeyboardInterrupt:
        print("Saving before quit...")
        save_model(model, "%sepoch_%s_loss_%.4f_time_%s.ckpt" %
                   ("./checkpoints/KeyboardInterrupt_", epoch, loss_meter.value()[0], 
                    time.strftime("%Y%m%d%H%M", time.localtime())))

if __name__== "__main__":
    argparser = argparse.ArgumentParser()
#    argparser.add_argument("--root", type=str, default="E:\\YanYi\\LSDroid\\MYDroid\\2012\\image")
    argparser.add_argument("--root", type=str, default="/home/zhangxin/MyDroid/Dataset/2012/image")
    argparser.add_argument("--data_pickle", type=str, default="")
    argparser.add_argument("--n_epochs", type=int, default=100)
    argparser.add_argument("--print_every", type=int, default=64)
    argparser.add_argument("-lr", type=float, default=0.001)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--load_model_path", type=str, default="")
    args = argparser.parse_args()
    print(vars(args))
    
    train(**vars(args))