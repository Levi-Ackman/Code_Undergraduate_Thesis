import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.FC_MLP import FC_MLP
from models.layers.ResNet18 import ResNet18
from models.layers.VIT16 import VIT16

class Model(nn.Module):
    def __init__(self,
                 configs,
                 ):
        super(Model, self).__init__()
        self.mri_encoder=ResNet18(configs.f_dim,configs.pretrain,configs.finetune) if configs.enc_mri=='resnet18' else VIT16(configs.f_dim,configs.pretrain,configs.finetune)
        self.fc_encoder=FC_MLP(configs.f_dim)
        self.fc_proj=nn.Sequential(nn.Linear(configs.f_dim, configs.l_dim),nn.Softmax(dim=-1))
        self.mri_proj=nn.Sequential(nn.Linear(configs.f_dim, configs.l_dim),nn.Softmax(dim=-1))
        self.temperature=configs.temperature
                                    
    def forward(self, mri_data,fc_data,label):
        mri_f=self.mri_proj(self.mri_encoder(mri_data))
        fc_f=self.fc_proj(self.fc_encoder(fc_data))
        contra_loss = self.contrastive_loss(mri_f, fc_f,self.temperature)
        y_hat=fc_f+mri_f
        cls_loss=F.mse_loss(F.softmax(y_hat,dim=-1),label)
        _,y_hat=torch.max(y_hat, 1)
        _,label=torch.max(label, 1)
        loss=contra_loss+cls_loss
        return y_hat,label,loss
    
    def contrastive_loss(self,mri_f, fc_f, temperature=1.0):
        mri_f = F.normalize(mri_f, p=2, dim=1)
        fc_f = F.normalize(fc_f, p=2, dim=1)
        sim = torch.matmul(mri_f, fc_f.T) / temperature
        batch_size = mri_f.size(0)
        labels = torch.arange(batch_size).to(mri_f.device)
        pos_loss = F.cross_entropy(sim, labels)
        neg_loss = F.cross_entropy(sim.T, labels)
        total_loss = pos_loss + neg_loss
        return total_loss



        
        
        