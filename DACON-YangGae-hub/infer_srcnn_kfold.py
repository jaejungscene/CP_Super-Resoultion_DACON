import os
from args import get_args_parser
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = get_args_parser().parse_args().cuda

import random
import pandas as pd
import numpy as np
import os
import cv2
import zipfile
from datetime import datetime

from srcnn import create_model
from utils import save_checkpoint, seed_everything
from dataset import CustomDataset, get_test_transform, get_train_transform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models
from torchvision import transforms

import warnings
warnings.filterwarnings(action='ignore')



def kfold_inference(model_list, test_loader):
    for m in model_list:
        # m.cuda()
        m.eval()
    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for lr_img, file_name in tqdm(iter(test_loader)):
            lr_img = lr_img.float().cuda()

            for i, m in enumerate(model_list):
                if i==0:
                    pred_hr_img = m(lr_img)
                else:
                    pred_hr_img += m(lr_img)
            
            pred_hr_img /= len(model_list)

            for pred, name in zip(pred_hr_img, file_name):
                pred = pred.cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = pred*255. ##################### float에 255를 곱한 후 uint8로 바꿈
                pred_img_list.append(pred.astype('uint8'))
                name_list.append(name)
    return pred_img_list, name_list


def run(args):
    test_df = pd.read_csv('./data/test.csv')
    test_dataset = CustomDataset(test_df, get_test_transform(), False, args)
    test_loader = DataLoader(   test_dataset, batch_size = args.batchSize,
                                shuffle=False, num_workers=args.workersNum  )
    
    model_list = []
    for i in range(5):
        model = create_model(args.model)
        model_state = torch.load(f'/home/ljj0512/private/DACON-YangGae-hub/log/2022-09-08 07:38:53/model{i}.pth.tar')
        model.load_state_dict(model_state)
        model_list.append(model)

    pred_img_list, pred_name_list = kfold_inference(model_list, test_loader)\

    os.makedirs(f'./submission/{args.expname}', exist_ok=True)
    os.chdir(f'./submission/{args.expname}')
    sub_imgs = []
    print("------- make submission -------")
    for path, pred_img in tqdm(zip(pred_name_list, pred_img_list)):
        cv2.imwrite(path, pred_img)
        sub_imgs.append(path)

    with zipfile.ZipFile("./submission.zip","w") as submission:
        for path in sub_imgs:
            submission.write(path)
    print("======== finish submission file =======")



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)