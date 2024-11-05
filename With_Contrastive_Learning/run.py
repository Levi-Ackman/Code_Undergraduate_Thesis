import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import time

fix_seed = 2424
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodality classification of Autism')

    # basic config
    parser.add_argument('--model', type=str, required=False, default='CAA',
                        help='model name')
    parser.add_argument('--num_folds', type=int, default=5, help='k_folds validation')

    # data loader
    parser.add_argument('--root_dir', type=str, default='../abide', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # model 
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrained or not')
    parser.add_argument('--finetune', type=bool, default=True, help='finetune whole encoder backbone or not')
    parser.add_argument('--enc_mri', type=str, default='vit16', help='type of mri encoder, choose from resnet18 or vit16')
    parser.add_argument('--l_dim', type=int, default=2, help='dimension of label')
    parser.add_argument('--f_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--temperature', type=int, default=1.0, help='temperature parameter for contrastive learning')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=24, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    for tem in [0.125,0.25,1.0,4.0,8.0]:
        for enc_mri in ['resnet18','vit16']:
            for pretrain in [True, False]:
                for finetune in [True, False]:
                    args.temperature=tem
                    args.enc_mri=enc_mri
                    args.pretrain=pretrain
                    args.finetune=finetune
                    print('Args in experiment:')
                    print(args)
                    Exp = Exp_Main
                    # setting record of experiments
                    setting = 'enc_mri_{}_f_dim_{}_temperature_{}_pretrain_{}_finetune_{}'.format(
                                args.enc_mri,
                                args.f_dim,
                                args.temperature,
                                args.pretrain,
                                args.finetune)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.kf_train(setting)
                    torch.cuda.empty_cache()