import torch
from skimage import measure
from torch.utils.data import DataLoader
from data_loader import TestDataset
from statistics import mean
from model import Student,Teacher
from sklearn.metrics import roc_auc_score , auc
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from torch.nn import functional as F
from numpy import ndarray


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

# 计算异常图
def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


# def get_ano_map(feature1,feature2):
#     mseloss=nn.MSELoss(reduction='none')
#     mse=mseloss(feature1,feature2)
#     mse_map=torch.mean(mse,dim=1)
#     cos=nn.functional.cosine_similarity(feature1,feature2,dim=1)
#     ano_map=torch.ones_like(cos)-cos
#     loss=(ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
#     mse_loss=(mse_map.view(mse_map.shape[0],-1).mean(-1)).mean()
#
#     return ano_map.unsqueeze(1) , loss , mse_map.unsqueeze(1), mse_loss


def test(obj_name, ckp_dir, data_dir, reshape_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    teacher=Teacher()
    teacher.to(device)

    student=Student()
    student.load_state_dict(torch.load(str(ckp_dir), map_location='cpu'))
    student.to(device)


    teacher.eval()
    student.eval()


    test_dataset = TestDataset(root_dir=data_dir, obj_name=obj_name, resize_shape=reshape_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    scores = []
    labels = []
    gt_list_px = []
    pr_list_px = []
    aupro_list = []

    with torch.no_grad():
        for idx, sample_test in enumerate(test_loader):
            image, label, gt = sample_test["image"], sample_test["label"], sample_test["gt_mask"]

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            img = image.to(device)
            efeature1, efeature2, efeature3 = teacher(img)
            inputs = [efeature1, efeature2, efeature3]

            dfeature1, dfeature2, dfeature3 = student(efeature1, efeature2, efeature3)
            outputs = [dfeature1, dfeature2, dfeature3]
            # 异常图的生成改成原来的模式
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            score = np.max(anomaly_map.ravel().tolist())

            scores.append(score)
            labels.append(label.numpy().squeeze())


    auroc_img = round(roc_auc_score(np.array(labels), np.array(scores)), 3)
    auroc_pix = round(roc_auc_score(np.array(gt_list_px), np.array(pr_list_px)), 3)
    return  auroc_img, auroc_pix, round(np.mean(aupro_list),3)



