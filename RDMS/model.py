import torch
import torch.nn as nn
from torchvision.models import resnet50,wide_resnet50_2
import torchvision.models as models
from sspcab_torch import SSPCAB

import torchvision


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(ConvTransposeBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.ConvTranspose2d(in_channel, F1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.ConvTranspose2d(in_channel, F3, kernel_size=2, stride=2, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

# model=resnet50()
# print(model.layer1)


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X



class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1 = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).conv1
        self.bn1 = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).bn1
        self.relu = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).relu
        self.maxpool = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).maxpool

        self.layer1 = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).layer1
        self.layer2 = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).layer2
        self.layer3=wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2).layer3


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # [1, 256, 64, 64]
        feature1 = x

        x = self.layer2(x)  # [1, 512, 32, 32]
        feature2 = x

        x=self.layer3(x)   #[1,1024,16,16]

        feature3=x


        return feature1, feature2,feature3

class Concat_T_S(nn.Module):
    def __init__(self):
        super(Concat_T_S,self).__init__()
        self.feature1con=nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1,bias=False),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU(inplace=True))
        self.feature2con=nn.Sequential(nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1,bias=False),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU(inplace=True))
        self.selayer=SSPCAB(channels=1024,kernel_dim=1,dilation=1)
        self.layer=nn.Sequential(
            IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
            IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
            IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
        )

    def forward(self,feature1,fearute2,feature3):
        feature1=self.feature1con(feature1)
        fearute2=self.feature2con(fearute2)
        x=feature1+fearute2+feature3
        x=self.selayer(x)
        x=self.layer(x)
        return x


class Student1(nn.Module):
    def __init__(self):
        super(Student1,self).__init__()
        self.layer2 = nn.Sequential(ConvTransposeBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512])
                                    )
        self.layer1 = nn.Sequential(ConvTransposeBlock(in_channel=512, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256])
                                    )

    def forward(self,x):
        x=self.layer2(x)
        x=self.layer1(x)
        feature1=x   #[1,256,64,64]

        return feature1

class Student2(nn.Module):
    def __init__(self):
        super(Student2,self).__init__()

        self.layer2 = nn.Sequential(ConvTransposeBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512])
                                    )
    def forward(self,x):

        x=self.layer2(x)
        feature2=x  #[1,512,32,32]

        return feature2

class Student3(nn.Module):
    def __init__(self):
        super(Student3,self).__init__()
        self.layer3 = nn.Sequential(IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512, 1024, 1024]),
                                    )

    def forward(self,x):
        feature3=self.layer3(x) # [1,1024,16,16]

        return feature3

class Student(nn.Module):
    def __init__(self):
        super(Student,self).__init__()
        self.concat=Concat_T_S()
        self.student1=Student1()
        self.student2=Student2()
        self.student3=Student3()

    def forward(self,efeature1,sfeature2,efeature3):
        x=self.concat(efeature1,sfeature2,efeature3)
        feature1=self.student1(x)
        feature2=self.student2(x)
        feature3=self.student3(x)

        return feature1, feature2, feature3

