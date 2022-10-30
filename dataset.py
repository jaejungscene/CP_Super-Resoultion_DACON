import cv2
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

## CustomDataset
class CustomDataset(Dataset):
    def __init__(self, df, transforms, train_mode, args):
        self.df = df
        self.transforms = transforms
        self.train_mode = train_mode
        self.args = args

    def __getitem__(self, index):
        lr_path = self.df['LR'].iloc[index]
        lr_img = cv2.imread(lr_path)
        
        if self.args.model == "srcnn":
            # interpolation으로 low resolution을 high resolution으로 size로 바꿈
            lr_img = cv2.resize(lr_img, (self.args.imgSize, self.args.imgSize), interpolation=cv2.INTER_CUBIC)

        if self.train_mode:
            hr_path = self.df['HR'].iloc[index]
            hr_img = cv2.imread(hr_path)
            if transforms is not None:
                transformed = self.transforms(image=lr_img, label=hr_img)
                lr_img = transformed['image'] / 255.
                hr_img = transformed['label'] / 255.
            return lr_img, hr_img
        else:
            file_name = lr_path.split('/')[-1]
            if transforms is not None:
                transformed = self.transforms(image=lr_img)
                lr_img = transformed['image'] / 255.
            return lr_img, file_name
        
    def __len__(self):
        return len(self.df)

def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )

def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )