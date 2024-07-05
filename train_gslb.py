import torch
from torch import nn

# !/user/bin/python
# coding=utf-8
# train_root = "/data/private/zhoucaixia/workspace/UD_Edge/"
# train_root = "F:\\experiments\\UAED-main"  # 当前项目所在的源路径(暂时用不上)
import os, sys
from statistics import mode

# sys.path.append(train_root)  # 好像用不上

import numpy as np
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib

matplotlib.use('Agg')

from data.data_loader_one_random_uncert import BSDS_RCFLoader, BipedDataset, TestDataset, NYUD_Loader

MODEL_NAME = "models.deeplab"
import importlib

Model = importlib.import_module(MODEL_NAME)

from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
import random
import numpy
from torch.autograd import Variable
import ssl
import cv2

ssl._create_default_https_context = ssl._create_unverified_context
from torch.distributions import Normal, Independent
from losses import cats_loss, rcf_loss, rcf_mask3_loss
from torch.utils.tensorboard import SummaryWriter
from models.u2net import u2net_full
import gc


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--LR', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=10, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
# 训练或测试时模型得到的模型参数文件以及其他一些文件存放源路径
parser.add_argument('--tmp', help='tmp folder', default='./result-NYUDv2/baseline(rcf+rtv1+side)all-size)')
# UAED_BSDS所在路径
parser.add_argument('--dataset', help='root folder of dataset', default=r'Z:\yx\experiments\dataset\NYUDv2')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--std_weight', default=1, type=float, help='weight for std loss')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
parser.add_argument('--resume', default=False, type=bool, help="use previous trained data")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR,
               args.tmp + "{}_{}_weightedstd{}_declr_adaexp".format(MODEL_NAME[7:], args.distribution, args.std_weight))
TMP_DIR = args.tmp  # TMP_DIR就是指模型在运行过程中得到的结果保持在哪里，包括模型参数，边缘检测图片等
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

file_name = os.path.basename(__file__)
# copyfile(join(train_root, MODEL_NAME[:6], MODEL_NAME[7:] + ".py"), join(TMP_DIR, MODEL_NAME[7:] + ".py"))
# copyfile(join(train_root, "train", file_name), join(TMP_DIR, file_name))
# copyfile(A, B)将A目录下的文件复制到B中
copyfile(join(MODEL_NAME[:6], MODEL_NAME[7:] + ".py"), join(TMP_DIR, MODEL_NAME[7:] + ".py"))
copyfile(join(file_name), join(TMP_DIR, file_name))
random_seed = 555
if random_seed > 0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)


def cross_entropy_loss_RCF(prediction, labelef):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    num_two = torch.sum((mask == 2).float()).float()
    assert num_negative + num_positive + num_two == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    assert num_two == 0
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    # new_mask = mask * torch.exp(std * ada)
    cost = F.binary_cross_entropy(
        prediction.float(), labelef.float(), weight=mask.detach(), reduction='sum')

    return cost


def step_lr_scheduler(optimizer, epoch, init_lr=args.LR, lr_decay_epoch=3):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def main():
    args.cuda = True
    # ----------------BSDS500------------
    # train_dataset = BSDS_RCFLoader(root=args.dataset, split="train")
    # test_dataset = BSDS_RCFLoader(root=args.dataset, split="test")
    # ----------------BIPED-------------
    # train_dataset = BipedDataset(data_root=args.dataset)
    # test_dataset = TestDataset(data_root=args.dataset)
    # ----------------NYUD--------------
    train_dataset = NYUD_Loader(root=args.dataset, split='train', setting=['image'])  # ['hha']
    test_dataset = NYUD_Loader(root=args.dataset, split='test', setting=['image'])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=8, drop_last=True, shuffle=False)
    # with open(join(args.dataset, "test.lst"), 'r') as f:
    #     test_list = f.readlines()
    # ----------------NYUD--------------
    with open(join(args.dataset, "image-test.txt"), 'r') as f:
        test_list = f.readlines()
    # test_list = [split(i.rstrip())[1] for i in test_list]  # BSDS
    test_list = [split(i.rstrip().split(" ")[0])[1] for i in test_list] # PIBED
    # assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    # model = Model.Mymodel(args).cuda()  # TANet, UAED
    # model = u2net_full().cuda()
    model = Model.Deeplabv3_res50().cuda()
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('Adam', args.LR)))
    sys.stdout = log
    writer = SummaryWriter("logs")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    with open(join(TMP_DIR, "loss-vlaue.txt"), 'w') as f:
        f.write("epoch loss-value\n")
    if args.resume:
        checkpoints_path = "result/baseline(rcf+side)-RTV-images/epoch-0-training-record/epoch-0-checkpoint.pth"
        model.load_state_dict(torch.load(checkpoints_path, map_location='cuda')['state_dict'])
    for epoch in range(args.start_epoch, args.maxepoch):
        # if epoch==0:
        #     test(model, test_loader, epoch=epoch, test_list=test_list,
        #     save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        loss_avg, batch_time = train(train_loader, model, optimizer, epoch, writer,
              save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        # print("----------------单尺度测试----------------")
        # test(model, test_loader, epoch=epoch, test_list=test_list,
        #      save_dir=join(TMP_DIR, 'epoch-%d-testing-record-SS' % epoch))
        # print("----------------多尺度测试---------------")
        # multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
        #                 save_dir=join(TMP_DIR, 'epoch-%d-testing-MS' % epoch))
        log.flush()  # write log
        writer.add_scalar("loss", loss_avg, epoch+1)
        with open(join(TMP_DIR, r"loss-vlaue.txt"), 'a') as f:
            f.write(f"{epoch+1} {loss_avg}\n")
        print(f"epoch:{epoch} was done and uses [{batch_time:.3f}] minutes")
        torch.cuda.empty_cache()
    writer.close()


def train(train_loader, model, optimizer, epoch, writer, save_dir):
    optimizer = step_lr_scheduler(optimizer, epoch)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    counter = 0
    # for i, (image, label, rtv_weight) in enumerate(train_loader):
    for i, (image, label, rtv_weight) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label, rtv_weight = image.cuda(), label.cuda(), rtv_weight.cuda()
        # image, label = image.cuda(), label.cuda()
        mean = model(image)
        counter += 1

        loss = rcf_mask3_loss(mean, label, rtv_weight)
        # loss = rcf_loss(mean, label)

        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        losses.update(loss, image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            torchvision.utils.save_image(1 - mean, join(save_dir, "iter-%d_mean.jpg" % i))

        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))
    del image, label, rtv_weight, mean, loss
    gc.collect()
    return losses.avg, batch_time.avg * len(train_loader)/60


def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        mean = model(image)
        mean = torch.sigmoid(mean)
        _, _, H, W = image.shape

        filename = splitext(test_list[idx])[0]

        mean = torch.squeeze(mean.detach()).cpu().numpy()
        # result_mean = np.zeros((H + 1, W + 1))  # 为什么会有这一步？
        result_mean = mean
        result_mean_png = Image.fromarray((result_mean * 255).astype(np.uint8))
        mean_save_dir = os.path.join(save_dir, "edges")
        mat_save_dir = os.path.join(save_dir, "edges-mat")

        if not os.path.exists(mean_save_dir):
            os.makedirs(mean_save_dir)
        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_mean_png.save(join(mean_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename), {"result": result_mean}, do_compression=True)


def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.6, 1, 1.6]
    for idx, image in enumerate(test_loader):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))

            mean = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            mean = torch.sigmoid(mean)
            mean = torch.squeeze(mean.detach()).cpu().numpy()
            fuse = cv2.resize(mean, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result = np.zeros((H + 1, W + 1))
        result[1:, 1:] = multi_fuse
        filename = splitext(test_list[idx])[0]

        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir = os.path.join(save_dir, "edges")
        mat_save_dir = os.path.join(save_dir, "edges-mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


if __name__ == '__main__':
    main()
