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
import wandb

from utils import save_checkpoint, seed_everything
from dataset import CustomDataset, get_test_transform, get_train_transform
from srgan import Generator
from srcnn import SRCNN
from myrcan import RCAN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models
from torchvision import transforms

import warnings
warnings.filterwarnings(action='ignore')

"""
if you want to use wandb, write "--wandb 1" in command

* srcnn *
python train.py --model srcnn --imgSize 2048 --epochs 30 --workersNum 4 --batchSize 12 --lr 1e-4 --foldNum 5 --cuda 0,1

* pretrained srresnet *
python train.py --model srresnet --epochs 30 --workersNum 4 --batchSize 12 --lr 1e-4 --cuda 0,1,2,3

* rcan *
python train.py --model myrcan --epochs 50 --workersNum 4 --batchSize 6 --lr 1e-4 --cuda 0,1,2,3,4,5
"""

# CFG = {
#     'IMG_SIZE':2048,
#     'EPOCHS':30,
#     'LEARNING_RATE':1e-4,
#     'BATCH_SIZE':12,
#     'SEED':41
# }


def validate(model, dl_valid, criterion):
    model.eval()
    valid_loss = []
    psnr = 0.0
    for i, (lr, target) in enumerate(dl_valid):
        target = target.cuda()
        output = model(lr)
        loss = criterion(output, target)
        psnr += peak_signal_noise_ratio(
            ((output.cpu().clone().detach().numpy())*255).astype("uint8"), 
            ((target.cpu().clone().detach().numpy())*255).astype("uint8")
            # output.cpu().clone().detach().numpy(), 
            # target.cpu().clone().detach().numpy()
        )
        valid_loss.append(loss.item())
    
    psnr /= len(valid_loss)
    _valid_loss = np.mean(valid_loss)
    return _valid_loss, psnr



def train(model, train_loader, optimizer, criterion, scheduler, args, dl_valid=None, idx=None):
    best_model = None
    best_loss = 9999
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        train_psnr = 0.0
        for lr_img, hr_img in tqdm(iter(train_loader)):
            lr_img, hr_img = lr_img.float().cuda(), hr_img.float().cuda()
            
            optimizer.zero_grad()
            
            pred_hr_img = model(lr_img)
            loss = criterion(pred_hr_img, hr_img)
            loss.backward()
            optimizer.step()
            
            train_psnr += peak_signal_noise_ratio(
                ((pred_hr_img.cpu().clone().detach().numpy())*255).astype("uint8"), 
                ((hr_img.cpu().clone().detach().numpy())*255).astype("uint8")
            )
            train_loss.append(loss.item())
                
        if scheduler is not None:
            scheduler.step()    
        _train_loss = np.mean(train_loss)

        train_psnr = train_psnr/len(train_loss)

        if args.foldNum <= 1: 
            print(f'Epoch [{epoch}]/[{args.epochs}] | Train Loss : [{_train_loss:.5f}] | Train PSNR : [{train_psnr:.5f}]')
            if args.wandb == True:
                wandb.log({'train psnr':train_psnr, 'train loss':_train_loss})
            if best_loss > _train_loss:
                best_loss = _train_loss
                best_model = model
                save_checkpoint(model, args)
        else: # kFold
            valid_loss, valid_psnr = validate(model, dl_valid, criterion)
            print(f'Epoch [{epoch}]/[{args.epochs}] | Fold : [{idx+1}] / {args.foldNum} | Train Loss : [{_train_loss:.5f}] | Train PSNR : [{train_psnr:.5f}]')
            print(f'Epoch [{epoch}]/[{args.epochs}] | Fold : [{idx+1}] / {args.foldNum} | Valid Loss : [{valid_loss:.5f}] | Valid PSNR : [{valid_psnr:.5f}]')
            if args.wandb == True:
                wandb.log({'validation loss':valid_loss, 'validation psnr':valid_psnr, 'train psnr':train_psnr, 'train loss':_train_loss})
            if best_loss > valid_loss:
                best_loss = valid_loss
                best_model = model
                save_checkpoint(model, args, filename=f"model{idx+1}.pth.tar")
    
    return best_model



def Kfold_train(args, train_df, criterion):
    kf = KFold(n_splits=args.foldNum)
    best_model_list = []
    for idx, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        print(f"---------------- Starting FOLD : {idx+1} / {args.foldNum} --------------")
        cs_train, cs_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]

        cs_train.reset_index(inplace=True)
        cs_train.drop('index', axis = 1, inplace = True)
        cs_valid.reset_index(inplace=True)
        cs_valid.drop('index', axis = 1, inplace = True)

        ds_train = CustomDataset(df = cs_train, train_mode = True, transforms = get_train_transform(), args=args)
        ds_valid = CustomDataset(df = cs_valid, train_mode = True, transforms = get_train_transform(), args=args)
        dl_train  = DataLoader(ds_train, batch_size = args.batchSize, shuffle=True)
        dl_valid = DataLoader(ds_valid, batch_size = args.batchSize, shuffle=False)

        model = create_model(args.model)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_model = train(model, dl_train, optimizer, criterion, scheduler, args, dl_valid=dl_valid, idx=idx)
        best_model_list.append(best_model)
    
    return best_model_list



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
                pred_img_list.append(pred.astype('uint8'))
                name_list.append(name)
    return pred_img_list, name_list



def kfold_inference(model_list, test_loader):
    for m in model_list:
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


def create_model(model_name):
    if model_name == "srcnn":
        model = SRCNN()
    elif model_name == "srresnet":
        pretrained_model_path = "/home/ljj0512/private/DACON-YangGae-hub/data/SRResNet_x4-ImageNet-2096ee7f.pth.tar"
        # pretrained_model_path = "/home/ljj0512/private/DACON-YangGae-hub/log/2022-09-10 11:22:23/model.pth.tar"
        model = Generator()
        print("Check whether to load pretrained model weights...")
        if pretrained_model_path:
            # Load checkpoint model
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            # Load model state dict. Extract the fitted model weights
            model_state_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                    k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
            # Overwrite the model weights to the current model
            model_state_dict.update(state_dict)
            model.load_state_dict(model_state_dict)
            print(f"Loaded `{pretrained_model_path}` pretrained model weights successfully.")
        else:
            print("Pretrained model weights not found.")
    elif model_name == "myrcan":
        model = RCAN()
    return nn.DataParallel(model).cuda()


def run(args):
    seed_everything(args.seed)

    train_df = pd.read_csv('./data/train.csv')
    criterion = nn.MSELoss().cuda()

    if args.foldNum <= 1:
        # train data load
        train_dataset = CustomDataset(train_df, get_train_transform(), True, args)
        train_loader = DataLoader(  train_dataset, batch_size = args.batchSize,
                                    shuffle=True, num_workers=args.workersNum   )
        model = create_model(args.model)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        best_model = train(model, train_loader, optimizer, criterion, scheduler, args)    

    else: # k-fold
        best_model = Kfold_train(args, train_df, criterion)

    # test data load
    test_df = pd.read_csv('./data/test.csv')
    test_dataset = CustomDataset(test_df, get_test_transform(), False, args)
    test_loader = DataLoader(   test_dataset, batch_size = args.batchSize,
                                shuffle=False, num_workers=args.workersNum  )
    
    # inference for test data
    pred_img_list, pred_name_list = inference(best_model, test_loader)\
        if not isinstance(best_model, list) else kfold_inference(best_model, test_loader)

    ## Submission
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
    # submission = zipfile.ZipFile("../submission.zip", 'w')
    # for path in sub_imgs:
    #     submission.write(path)
    # submission.close()
    # print('Done.')




if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args.expname)
    print("model :",args.model)
    print("lr :",args.lr)
    print("batch size :",args.batchSize)
    if args.wandb == True:
        wandb.init(project='YangGae-hub-SR', name=args.model+args.expname, entity='jaejungscene')
    run(args)
