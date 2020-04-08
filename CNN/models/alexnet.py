# coding:utf8
import torch
from torch import nn
from .basic_module import BasicModule


class AlexNet(BasicModule):
    """
    根据图片尺寸修改的AlexNet
    """

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.name = 'AlexNet'
        self.features = nn.Sequential(                         # (1, 800, 4)
            nn.Conv2d(1, 64, kernel_size=4),                   # (64, 797, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (64, 266, 1)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),      # (192, 266, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (192, 89, 1)
            nn.Conv2d(192, 384, kernel_size=5, padding=2),     # (384, 89, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=5, padding=2),     # (256, 89, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),     # (256, 89, 1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # (256, 30, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))            # (256, 6, 6)
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__== "__main__":
    f = open("F:\\test\\decompileDataset\\image\\com.just4fun.spiderinphone_0.pickle", "rb")
    image = torch.load(f)
    image = torch.unsqueeze(image, dim=0).float()
    image = torch.unsqueeze(image, dim=0).float()
    print(image.shape)
    input = torch.zeros(1, 1, 800, 4).float()
    print(input.shape)
    model_1 = AlexNet()
    output = model_1(image)
    print(output.shape)