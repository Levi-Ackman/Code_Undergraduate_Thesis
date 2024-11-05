import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import ABIDE_Data
from torch.utils.data import WeightedRandomSampler,SubsetRandomSampler,DataLoader
from sklearn.model_selection import KFold
from exp.exp_basic import Exp_Basic
from models import CAA
from utils.tools import EarlyStopping
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
import pandas as pd 
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'CAA': CAA
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return model

    def _get_data(self):
        dataset = ABIDE_Data(root_dir=self.args.root_dir)
        # 定义交叉验证
        kf = KFold(n_splits=self.args.num_folds, shuffle=True)
        
        # 提前划分数据集为训练集和验证集的索引
        train_loaders = []
        vali_loaders = []
        for train_indices, val_indices in kf.split(dataset):
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=train_sampler)
            vali_loader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=val_sampler)

            train_loaders.append(train_loader)
            vali_loaders.append(vali_loader)
            
        return train_loaders, vali_loaders

    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        acc_count=0
        sample_cpunt=0
        with torch.no_grad():
            for _, (img_data,fc_data,label) in enumerate(vali_loader):
                img_data = img_data.float().to(self.device)
                fc_data = fc_data.float().to(self.device)
                label = label.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat,label,loss = self.model(img_data,fc_data,label)
                else:
                    y_hat,label,loss = self.model(img_data,fc_data,label)
                sample_cpunt+=img_data.size(0)
                acc_count+=(y_hat == label).sum().item()
                total_loss.append(loss.cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss,acc_count/sample_cpunt

    def train(self, train_loader, vali_loader,path):
        path = os.path.join(self.args.checkpoints, path)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            train_loss = []
            acc_count=0
            sample_cpunt=0
            self.model.train()
            epoch_time = time.time()
            for _, (img_data,fc_data,label) in enumerate(train_loader):
                model_optim.zero_grad()

                img_data = img_data.float().to(self.device)
                fc_data = fc_data.float().to(self.device)
                label = label.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat,label,loss = self.model(img_data,fc_data,label)
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                else:
                    y_hat,label,loss = self.model(img_data,fc_data,label)
                    loss.backward()
                    model_optim.step()
                sample_cpunt+=img_data.size(0)
                acc_count+=(y_hat == label).sum().item()
                train_loss.append(loss.cpu().detach().numpy())
                
            train_loss = np.average(train_loss)
            train_acc=acc_count/sample_cpunt
            vali_loss,vali_acc = self.vali(vali_loader)

            print("Epoch: {0}, | Train Loss: {1:.3f} Vali Loss: {2:.3f}".format(epoch + 1, train_loss, vali_loss))
            print("Epoch: {0}, | Train Acc: {1:.4f} Vali Acc: {2:.4f}".format(epoch + 1, train_acc, vali_acc))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return 
    
    def test(self, vali_loader):
        self.model.eval()
        y_hats=[]
        labels=[]
        with torch.no_grad():
            for _, (img_data,fc_data,label) in enumerate(vali_loader):
                img_data = img_data.float().to(self.device)
                fc_data = fc_data.float().to(self.device)
                label = label.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat,label,loss = self.model(img_data,fc_data,label)
                else:
                    y_hat,label,loss = self.model(img_data,fc_data,label)
                labels.append(label.cpu().numpy())
                y_hats.append(y_hat.cpu().numpy())
        if len(y_hats)>0:
            y_hats = np.concatenate(y_hats, axis=0)    
            labels = np.concatenate(labels, axis=0)    
        else:
            y_hats=y_hats[0]
            labels=labels[0]
        return self.evaluate_model(labels, y_hats)
    def evaluate_model(self,labels, y_hats):
        # 计算混淆矩阵
        cm = confusion_matrix(labels, y_hats)
        # 计算评估指标
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        acc_clas_0 = TN / (TN + FP)
        acc_clas_1 = TP / (TP + FN)
        # 打印评估矩阵
        print("Confusion Matrix:")
        df_cm = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Predicted 0", "Predicted 1"])
        print(df_cm)
        print("\nAccuracy:{0:4f}".format(accuracy))
        print("acc_clas_0:{0:4f}" .format(acc_clas_0))
        print("acc_clas_1:{0:4f}".format(acc_clas_1))
        
        return accuracy, acc_clas_0, acc_clas_1

    def kf_train(self, setting):
        train_loaders,vali_loaders=self._get_data()
        avg_accuracy,avg_acc_clas_0,avg_acc_clas_1=0.,0.,0.
        for fold, (train_loader, vali_loader) in enumerate(zip(train_loaders, vali_loaders)):
            print(f"Fold {fold + 1}/{self.args.num_folds}")
            self.train(train_loader, vali_loader, setting+str(fold + 1))
            accuracy, acc_clas_0, acc_clas_1=self.test(vali_loader)
            avg_accuracy+=accuracy
            avg_acc_clas_0+=acc_clas_0
            avg_acc_clas_1+=acc_clas_1
            torch.cuda.empty_cache()
        avg_accuracy=avg_accuracy/self.args.num_folds
        avg_acc_clas_0=avg_acc_clas_0/self.args.num_folds
        avg_acc_clas_1=avg_acc_clas_1/self.args.num_folds
        print('total avg vali acc:{0:4f}  total avg vali acc_clas_0: {1:4f}  total avg vali acc_clas_1: {2:4f}'
              .format(avg_accuracy,avg_acc_clas_0,avg_acc_clas_1))
        return 