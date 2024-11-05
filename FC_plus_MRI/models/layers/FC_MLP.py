import torch
import torch.nn as nn

class FC_MLP(nn.Module):
    def __init__(self,
                 f_dim=128,
                 ):
        super(FC_MLP, self).__init__()
        self.fc_proj=nn.Sequential(nn.Linear(19900, f_dim),nn.GELU(),nn.Linear(f_dim, f_dim))
    def forward(self, inp):
        # inp： [B,fc_dim/19900]
        inp=self.fc_proj(inp) # [B,fc_dim]=>[B,D]
        return inp
