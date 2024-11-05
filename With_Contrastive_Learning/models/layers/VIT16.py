import torch.nn as nn
from timm import create_model

class VIT16(nn.Module):
    def __init__(self, f_dim=256,pretrain=True,finetune=False):
        super(VIT16, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrain)
        for param in self.vit.parameters():
            param.requires_grad = finetune
        self.vit.head = nn.Linear(self.vit.head.in_features, f_dim)
        self.conv = nn.Conv2d(128, 3, kernel_size=33) 

    def forward(self, x):
        x=self.conv(x)
        features = self.vit(x)
        return features