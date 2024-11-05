import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import scipy
import numpy as np
import pandas as pd
class ABIDE_Data(Dataset):
    def __init__(self, root_dir):
        self.img_root_dir = root_dir+'/MRI'
        self.fc_root_dir = root_dir+'/fMRI'
        self.labels = pd.read_csv(root_dir+'/label.csv')['DX_GROUP']
        self.names = pd.read_csv(root_dir+'/label.csv')['SUB_ID']
        self.sub_names = [int(name) for name in os.listdir(self.fc_root_dir) if os.path.isdir(os.path.join(self.fc_root_dir, name))]
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.img_file_paths = self._get_img_file_paths()
        self.fc_file_paths = self._get_fc_file_paths()

    def __len__(self):
        return len(self.img_file_paths)

    def __getitem__(self, idx):
        img_file_path = self.img_file_paths[idx]
        fc_file_path = self.fc_file_paths[idx]

        # Load NIfTI file using nibabel
        img = nib.load(img_file_path)
        img_data = img.get_fdata()
        img_data = (img_data - 224) / 224
        # Apply transformations (if needed)
        img_data = self.transform(img_data).permute(1,0,2)
        fc_data=scipy.io.loadmat(fc_file_path)['connectivity']
        fc_data=self._fc_feature(fc_data)
        label=torch.tensor([0,1]) if np.array(self.labels.iloc[self.names.tolist().index(self.sub_names[idx])] == 2) else torch.tensor([1,0]) 
        return img_data,fc_data,label

    def _get_img_file_paths(self):
        img_file_paths = []
        for subdir, dirs, files in os.walk(self.img_root_dir):
            for file in files:
                if file.endswith(".nii"):
                    img_file_paths.append(os.path.join(subdir, file))
        return img_file_paths
    def _get_fc_file_paths(self):
        fc_file_paths = []
        for subdir, dirs, files in os.walk(self.fc_root_dir):
            for file in files:
                if file.endswith(".mat"):
                    fc_file_paths.append(os.path.join(subdir, file))
        return fc_file_paths
    def _fc_feature(self, fc):
        # 提取上三角部分
        upper_triangular = np.triu(fc, k=1)
        # 转换为张量
        fc_tensor = torch.tensor(upper_triangular[np.nonzero(upper_triangular)])
        return fc_tensor
