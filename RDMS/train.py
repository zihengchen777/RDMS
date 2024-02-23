import torch
import torch.nn as nn
from data_loader import TrainDataset
from torch.utils.data import DataLoader
import argparse
import os
# from model import stage1,stage1_distillation,stage2_distillation
from model import Teacher , Student
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from loss import CosineLoss,get_ano_map,ssim
from test import test
import matplotlib.pyplot as plt



def train(obj_name,args):
    resize_shape=256
    print("start train {}".format(obj_name))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device=torch.device("cpu")

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    cur_time='{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    run_time=str(obj_name)+"_lr"+str(args.lr)+"_bs"+str(args.bs)+"_"+cur_time

    os.mkdir("./checkpoints/Res50/" + run_time)
    #os.mkdir("./checkpoints/Res50/stage2/" + run_time)

    # stage=stage1()
    # stage_distillation=stage1_distillation()
    # stage_distillation2=stage2_distillation()
    teacher=Teacher()
    student=Student()

    teacher.to(device)
    student.to(device)

    teacher.eval()

    train_dataset=TrainDataset(root_dir=args.data_path,obj_name=obj_name,resize_shape=resize_shape)
    train_loader=DataLoader(dataset=train_dataset,batch_size=args.bs,shuffle=True,drop_last=True,
                            num_workers=32,persistent_workers=True,pin_memory=True,prefetch_factor=2)

    cos_similarity = CosineLoss()

    optimizer = torch.optim.Adam(student.parameters(), betas=(0.5, 0.999), lr=args.lr)
    # optimizer = torch.optim.Adam([{"params":stage_distillation.parameters()},{"params":stage_distillation2.parameters()}],betas=(0.5,0.999),lr=args.lr)

    #cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-9)


    auroc_img_best, img_step = 0, 0
    auroc_pix_best, pix_step = 0, 0
    pro_best, pro_step = 0, 0
    train_loss = []
    step_eopch = []

    for step in tqdm(range(args.epochs),ascii=True):

        student.train()
        train_loss_total=0

        for idx,sample in enumerate(train_loader):
            images=sample['image'].to(device)

            efeature1, efeature2, efeature3 = teacher(images)
            #d_feature1, d_feature2 , du_feature1 ,du_feature2= stage_distillation(output)

            dfeature1, dfeature2, dfeature3=student(efeature1, efeature2, efeature3)
            loss1 = cos_similarity(efeature1,dfeature1)
            loss2 = cos_similarity(efeature2, dfeature2)
            loss3 = cos_similarity(efeature3, dfeature3)
            # cos_map1, cos_loss1, mse_map1,mse_loss1=get_ano_map(efeature1,dfeature1)
            # cos_map2, cos_loss2, mse_map2,mse_loss2=get_ano_map(efeature2, dfeature2) # 目前只让两个学生自主学习
            # cos_map3, cos_loss3, mse_map3,mse_loss3=get_ano_map(efeature3, dfeature3)


            # cos_map1=nn.functional.interpolate(cos_map1, size=(resize_shape, resize_shape), mode='bilinear',align_corners=True)
            # cos_map2 = nn.functional.interpolate(cos_map2, size=(resize_shape, resize_shape), mode='bilinear',align_corners=True)
            # mse_map3=nn.functional.interpolate(mse_map3, size=(resize_shape, resize_shape), mode='bilinear',align_corners=True)
            # mse_map4=nn.functional.interpolate(mse_map4, size=(resize_shape, resize_shape), mode='bilinear',align_corners=True)




            loss = loss1+loss2+loss3



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        step_eopch.append(int(step) + 1)
        train_loss.append(train_loss_total)

        if (int(step) + 1) % 20 == 0:
                plt.plot(step_eopch, train_loss, 'b o', label='train loss')
                plt.show()



        if (args.test_interval > 0) and (int(step) % args.test_interval == 0):
                ckp_path1 = str(args.checkpoint_path + "Res50/" + run_time + "/epoch" + str(step) + ".pth")
                # ckp_path2 = str(args.checkpoint_path + "Res50/stage2/" + run_time + "/epoch" + str(step) + ".pth")
                torch.save(student.state_dict(), ckp_path1)
                # torch.save(stage_distillation2.state_dict(),ckp_path2)
                auroc_img, auroc_pix,pro = test(obj_name=obj_name, ckp_dir=ckp_path1, data_dir=args.data_path,
                                                       reshape_size=resize_shape)
                # test_loss, auroc_img, auroc_pix = test(obj_name=obj_name, ckp_dir=ckp_path1, data_dir=args.data_path,
                #                                        reshape_size=resize_shape)

                # if auroc_img <= auroc_img_best and auroc_pix <= auroc_pix_best:
                #     os.remove(ckp_path1)

                if auroc_img >= auroc_img_best:
                    auroc_img_best = auroc_img
                    img_step = int(step)
                    print("img_step:{}, auroc_img_best:{}".format(img_step, auroc_img_best))
                if auroc_pix >= auroc_pix_best:
                    auroc_pix_best = auroc_pix
                    pix_step = int(step)
                    print("pix_step:{}, auroc_pix_best:{}".format(pix_step, auroc_pix_best))
                if pro>=pro_best:
                    pro_best=pro
                    pro_step=int(step)
                    print("pro_step:{},pro_best:{}".format(pro_step,pro_best))

    return run_time, auroc_img_best, auroc_pix_best, pro_best, img_step, pix_step,pro_step


def write2txt(filename, content):
    f = open(filename, 'a')
    f.write(str(content) + "\n")
    f.close()

parser = argparse.ArgumentParser()
    #parser.add_argument('--obj_id', action='store', type=int, required=True)
parser.add_argument('--bs', action='store', type=int, required=False, default=8)
parser.add_argument('--lr', action='store', type=float, required=False, default=0.0001)
parser.add_argument('--epochs', action='store', type=int, required=False, default=200)
parser.add_argument('--gpu_id', action='store', type=int, required=False, default=0)
parser.add_argument('--data_path', action='store', type=str, required=False, default="../Reverse_Disstilation/datasets/mvtec/")
parser.add_argument('--checkpoint_path', action='store', type=str, required=False, default="./checkpoints/")
    # parser.add_argument('--visualize', action='store_true')
parser.add_argument('--test_interval', action='store', type=int, required=False, default=5)

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
obj_names = ['bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper']

log_txt_name = "./logs_txt/" + str("{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now())) + ".txt"
os.open(log_txt_name, os.O_CREAT)  # os.mknod(log_txt_name)
    # os.mknod(log_txt_name)

write2txt(log_txt_name, "log title")

for i in range(15):
    obj_name = obj_names[i]
    model_name, auroc_img_best, auroc_pix_best, pro_best,img_step, pix_step ,pro_step= train(obj_name, args)
    write2txt(log_txt_name, str(model_name) + " || auroc_img: " + str(auroc_img_best) + " epoch:" + str(
    img_step) + " || auroc_pix: " + str(auroc_pix_best) + " epoch:" + str(pix_step)+" || pro: "+str(pro_best)+" eopch:"+str(pro_step))

# obj_name=obj_names[13]
# model_name, auroc_img_best, auroc_pix_best, img_step, pix_step = train(obj_name, args)
# write2txt(log_txt_name, str(model_name) + " || auroc_img: " + str(auroc_img_best) + " epoch:" + str(
# img_step) + " || auroc_pix: " + str(auroc_pix_best) + " epoch:" + str(pix_step))