import os
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from pathlib import Path
from torchvision import transforms
import numpy as np
# cable normalize([0.27288333, 0.3431031, 0.38527545], [0.16097097, 0.21066363, 0.22941259])

class TrainDataset(Dataset):
    def __init__(self, root_dir, obj_name, transform=None, resize_shape=None):
        self.root_dir = Path(root_dir)
        self.obj_name = obj_name
        self.resize_shape = resize_shape
        self.image_names = sorted(glob.glob(root_dir + '/' + obj_name + '/train/good/*.png'))
        # print(root_dir+obj_name+'/train/good/*.png')
        # print(self.image_names)
        # self.image_names=sorted(os.listdir(os.path.join(self.root_dir,self.obj_name+'/train/good')))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            #self.transform.transforms.append(transforms.CenterCrop(size=256))
            # self.transform.transforms.append(transforms.RandomHorizontalFlip())
            # self.transform.transforms.append(transforms.RandomVerticalFlip())
            # #
            # self.transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
            self.transform.transforms.append(transforms.ToTensor())
            # pill ([0.23901816, 0.23942311, 0.25671047], [0.2608697, 0.2609273, 0.24691993])
            # self.transform.transforms.append(transforms.Normalize([0.45665544, 0.47128767, 0.44821942], [0.11267109, 0.124385715, 0.11953287]))

            # ([0.72224903, 0.72224903, 0.72224903], [0.13175528, 0.13175528, 0.13175528])
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                       std=[0.229, 0.224, 0.225]))

            # self.transform.transforms.append(transforms.Normalize(mean=[0.722,0.722,0.722],
            #                                                       std=[0.132,0.132,0.132]))  # screw

            # self.transform.transforms.append(transforms.Normalize(mean=[0.578, 0.577, 0.577],
            #                                                       std=[0.116, 0.116, 0.116]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.385, 0.385, 0.385],
            #                                                       std=[0.146, 0.146, 0.146]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.272, 0.343, 0.385],
            #                                                       std=[0.160, 0.210, 0.229]))
          # ([0.3261558, 0.4142513, 0.4665436], [0.1523632, 0.21413396, 0.23781462])
        #([0.27288333, 0.3431031, 0.38527545], [0.16097097, 0.21066363, 0.22941259])
        # ([0.38491097, 0.38491097, 0.38491097], [0.14454708, 0.14454708, 0.14454708]) grid

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(str(self.image_names[idx])).convert("RGB")
        # img.show()
        img = self.transform(img)
        return {"image": img}


class TestDataset(Dataset):
    def __init__(self, root_dir, obj_name, transform=None, resize_shape=None):
        self.root_dir = Path(root_dir)
        self.obj_name = obj_name
        self.resize_shape = resize_shape
        self.image_names = sorted(glob.glob(root_dir +'/'+ self.obj_name + "/test/*/*.png"))
        self.gt_root = "../Reverse_Disstilation/" + "datasets/mvtec/" + self.obj_name + "/ground_truth/"

        if transform is not None:
            self.transform = transform
        else:
            # image preprocess
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            #self.transform.transforms.append(transforms.CenterCrop(size=256))
            # self.transform.transforms.append(transforms.RandomHorizontalFlip())
            # self.transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
            self.transform.transforms.append(transforms.ToTensor())
            # self.transform.transforms.append(transforms.Normalize(mean=[0.722, 0.722, 0.722],
            #                                                       std=[0.132, 0.132, 0.132]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.239, 0.239, 0.257],
            #                                                       std=[0.261, 0.261, 0.247]))
            # self.transform.transforms.append(
            #     transforms.Normalize([0.45665544, 0.47128767, 0.44821942], [0.11267109, 0.124385715, 0.11953287]))
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                       std=[0.229, 0.224, 0.225]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.578,0.578,0.578],
            #                                                       std=[0.116,0.116,0.116]))  # screw
            # self.transform.transforms.append(transforms.Normalize(mean=[0.385, 0.385, 0.385],
            #                                                       std=[0.146, 0.146, 0.146]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.578, 0.577, 0.577],
            #                                                       std=[0.116, 0.116, 0.116]))
            # self.transform.transforms.append(transforms.Normalize(mean=[0.272, 0.343, 0.385],
            #                                                       std=[0.160, 0.210, 0.229]))
            # screw ([0.5776494, 0.5776494, 0.5776494], [0.116197616, 0.116197616, 0.116197616])
            # gt preprocess
            self.gt_transform = transforms.Compose([])
            self.gt_transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            self.gt_transform.transforms.append(transforms.ToTensor())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_path = str(self.image_names[idx])
        label = img_path.split("/")[-2]
        gt_path = self.gt_root + label + "/" + img_path.split("/")[-1][:3] + "_mask.png"
        img = Image.open(img_path).convert("RGB")
        label = img_path.split("/")[-2]
        img = self.transform(img)

        if label == "good":
            gt_img = np.array([0], dtype=np.float32)
            gt_pix = torch.zeros([1, self.resize_shape, self.resize_shape])
        else:
            gt_img = np.array([1], dtype=np.float32)
            gt_pix = self.gt_transform(Image.open(gt_path))

        return {"image": img, "label": gt_img, "gt_mask": gt_pix}

        # good : 0, anomaly : 1

# train_data=TrainDataset(root_dir="../Reverse_Disstilation/datasets/mvtec",obj_name='bottle',resize_shape=256)
# print(train_data[10])
# print(len(train_data))

# test_data=TestDataset(root_dir="./datasets/mvtec",obj_name='bottle',resize_shape=256)
#
# print(test_data[10]['image'])
# print(test_data[10]['label'])
#
# print(test_data[10]['gt_mask'])


