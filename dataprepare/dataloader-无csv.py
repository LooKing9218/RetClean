# --coding:utf-8--
from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image
import os



def get_data_list(root):
    cls_dict = {
        'Normal': 0,
        'AMD': 1,
        'CSCR': 2,
        'DR': 3,
        'PM': 4,
        'RD': 5,
        'RP': 6,
        'RVO': 7
    }
    img_list = []
    for cls_fold in cls_dict.keys():
        img_files = os.listdir(os.path.join(root, cls_fold))
        for img_file in img_files:
            img_list.append(
                [
                    os.path.join(root, cls_fold, img_file),
                    int(cls_dict[cls_fold])
                ]
            )

    return img_list



class DatasetTrainSplitVal(Dataset):
    def __init__(self,data_list,mode = 'train'):

        self.data_list = data_list
        if mode == 'train':
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.299, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.299, 0.224, 0.225])
            ])

    def __getitem__(self,index):
        image_file,label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, label

    def __len__(self):
        return len(self.data_list)






class DatasetCFP(Dataset):
    def __init__(self,root,mode = 'train'):
        self.data_list = self.get_files(root)
        if mode == 'train':
            self.transforms= T.Compose([
                T.Resize((224,224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])

    def getSubFolders(self,folder):
        subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
        return sorted(subfolders)
    def get_files(self,root):
        # cls_folders = self.getSubFolders(root)
        cls_dict = {
            'Normal':0,
            'AMD':1,
            'CSCR':2,
            'DR':3,
            'PM':4,
            'RD':5,
            'RP':6,
            'RVO':7
        }

        img_list = []
        for cls_fold in cls_dict.keys():
            img_files = os.listdir(os.path.join(root,cls_fold))
            for img_file in img_files:
                img_list.append(
                    [
                        os.path.join(root, cls_fold,img_file),
                        int(cls_dict[cls_fold])
                    ]
                )

        return img_list

    def __getitem__(self,index):
        image_file,label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, label

    def __len__(self):
        return len(self.data_list)

