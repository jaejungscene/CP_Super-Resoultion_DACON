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

from utils import save_checkpoint, seed_everything
from dataset import CustomDataset, get_test_transform, get_train_transform
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from srgan import Generator

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models
from torchvision import transforms

import warnings
warnings.filterwarnings(action='ignore')


"""
python infer_srresnet.py --model srresnet --batchSize 12 --expname {your own name} --cuda 0,1,2,3
"""

def inference(model, test_loader):
    model.eval()
    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for lr_img, file_name in tqdm(iter(test_loader)):
            lr_img = lr_img.float().cuda()

            pred_hr_img = model(lr_img)

            for pred, name in zip(pred_hr_img, file_name):
                pred = pred.cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = pred*255. ##################### float에 255를 곱한 후 uint8로 바꿈
                pred_img_list.append(pred.astype('int32'))
                name_list.append(name)
    return pred_img_list, name_list


def run(args):
    test_df = pd.read_csv('./data/test.csv')
    test_dataset = CustomDataset(test_df, get_test_transform(), False, args)
    test_loader = DataLoader(   test_dataset, batch_size = args.batchSize,
                                shuffle=False, num_workers=args.workersNum  )
    
    model = create_model()
    print("Build SRResNet model successfully.")

    print("Check whether to load pretrained model weights...")
    pretrained_model_path = "/home/ljj0512/private/DACON-YangGae-hub/log/2022-09-10 11:22:23/model.pth.tar"
    if pretrained_model_path:
        # Load checkpoint model
        pre_state_dict = torch.load(pretrained_model_path)
        # Load model state dict. Extract the fitted model weights
        model.load_state_dict(pre_state_dict)
        print(f"Loaded `{pretrained_model_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    model = nn.DataParallel(model).cuda()

    pred_img_list, pred_name_list = inference(model, test_loader)

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


def create_model() -> nn.DataParallel:
    model = Generator()
    return model.cuda()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)