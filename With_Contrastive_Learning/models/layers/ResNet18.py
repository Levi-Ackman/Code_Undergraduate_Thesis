import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, f_dim=256,pretrain=True,finetune=False):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
        for param in self.resnet.parameters():
            param.requires_grad = finetune
        self.conv = nn.Conv2d(128, 3, kernel_size=33) 
        self.resnet.fc = nn.Linear(512, f_dim)
    def forward(self, x):
        x = self.conv(x)
        features = self.resnet(x)
        return features