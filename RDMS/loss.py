
import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss,self).__init__()

    def forward(self,feature1,feature2):
        cos=nn.functional.cosine_similarity(feature1,feature2,dim=1)
        ano_map=torch.ones_like((cos))-cos
        loss = (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        return loss


def get_ano_map(feature1,feature2):
    mseloss=nn.MSELoss(reduction='none')
    mse=mseloss(feature1,feature2)
    mse_map=torch.mean(mse,dim=1)
    cos=nn.functional.cosine_similarity(feature1,feature2,dim=1)
    ano_map=torch.ones_like(cos)-cos
    loss=(ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
    mse_loss=(mse_map.view(mse_map.shape[0],-1).mean(-1)).mean()

    return ano_map.unsqueeze(1) , loss , mse_map.unsqueeze(1), mse_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = (_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    #print(ssim_map)

    if size_average:
        #print("size_average running!")
        ssim_map=ssim_map.mean(dim=1)
        #ssim_map=torch.ones_like(ssim_map)-ssim_map
        # (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        # loss, ano_map
        return ssim_map.view(ssim_map.shape[0],-1).mean(-1).mean(), ssim_map.unsqueeze(1) # loss, map
    else:
        print("no running")
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# x=torch.rand([16,512,64,64],dtype=torch.float32)
#
# y=torch.rand([16,512,64,64],dtype=torch.float32)
# #
# ano_map , cos_loss , mse_map,mse_loss=get_ano_map(x,y)
# print(cos_loss)
# print(mse_loss)
#
# loss1,ssim_map=ssim(x,y)
# print(ssim_map.shape)