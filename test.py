import os
from torch.distributions import Independent, Normal
import torch
import numpy as np
from PIL import Image
import scipy
import cv2
import argparse
from tqdm import tqdm
from data.data_loader_one_random_uncert import BSDS_RCFLoader, TestDataset, NYUD_Loader
from models import sigma_logit_unetpp

parser = argparse.ArgumentParser(description="AED-TANet Testing")
parser.add_argument("--distribution", type=str, default="gs", help="the output distribution")
args = parser.parse_args()


def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    idx = 0
    pbar = tqdm(test_loader)
    for image in pbar:
        image = image.cuda()
        mean = model(image)
        mean = torch.sigmoid(mean)
        _, _, H, W = image.shape
        filename = os.path.splitext(test_list[idx])[0]
        idx += 1

        mean = torch.squeeze(mean.detach()).cpu().numpy()
        # BSDS500
        result_mean = np.zeros((H + 1, W + 1))  # 为什么会有这一步？
        result_mean[1:, 1:] = mean
        # BIPED
        # result_mean = mean[16:, :]
        # NYUD
        # assert mean.shape == (448, 576)
        # result_mean = cv2.resize(mean, dsize=(560, 425))
        # assert result_mean.shape == (425, 560)
        result_mean_png = Image.fromarray((result_mean * 255).astype(np.uint8))
        mean_save_dir = os.path.join(save_dir, "edges")
        mat_save_dir = os.path.join(save_dir, "edges-mat")

        # mean_sigmoid = torch.squeeze(mean_sigmoid).detach().cpu().numpy()
        # cv2.imshow("mean_sigmoid", mean_sigmoid)
        # cv2.waitKey()

        if not os.path.exists(mean_save_dir):
            os.makedirs(mean_save_dir)
        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_mean_png.save(os.path.join(mean_save_dir, "%s.png" % filename))
        scipy.io.savemat(os.path.join(mat_save_dir, "%s.mat" % filename), {"result": result_mean})

        # std = torch.squeeze(std.detach()).cpu().numpy()
        # result_std = np.zeros((H + 1, W + 1))
        # result_std[1:, 1:] = std
        # result_std_png = Image.fromarray((result_std * 255).astype(np.uint8))
        # std_save_dir = os.path.join(save_dir, "std")
        #
        # if not os.path.exists(std_save_dir):
        #     os.makedirs(std_save_dir)
        # result_std_png.save(os.path.join(std_save_dir, "%s.png" % filename))
    pbar.close()


def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.6, 1, 1.6]
    idx = 0
    pbar = tqdm(test_loader)
    for image in pbar:
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))

            mean = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            mean = torch.sigmoid(mean)
            # outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            # outputs = torch.sigmoid(outputs_dist.rsample())
            # mean = torch.clamp(mean, 0, 255)
            result = torch.squeeze(mean.detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result = np.zeros((H + 1, W + 1))
        result[1:, 1:] = multi_fuse
        filename = os.path.splitext(test_list[idx])[0]
        idx += 1
        result_png = Image.fromarray((result * 255).astype(np.uint8))
        # result_png = Image.fromarray((result).astype(np.uint8))
        png_save_dir = os.path.join(save_dir, "edges")
        mat_save_dir = os.path.join(save_dir, "edges-mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(os.path.join(png_save_dir, "%s.png" % filename))
        scipy.io.savemat(os.path.join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
    pbar.close()


def main():
    # 数据
    test_dataset = BSDS_RCFLoader(root="F:\\experiments\\dataset\\BSDS-UAED", split="test")
    # ----------------BIPED-------------
    # test_dataset = TestDataset(data_root=r"F:\experiments\dataset\archive\BIPED\BIPED")
    # NYUD
    # test_dataset = NYUD_Loader(root=r"F:\experiments\dataset\NYUD", split='test', setting=['image'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        num_workers=8, drop_last=True, shuffle=False)
    # BIPED, BSDS
    with open(os.path.join("F:\\experiments\\dataset\\BSDS-UAED", "test.lst"), 'r') as f:
        test_list = f.readlines()
    # NYUD
    # with open(os.path.join(r"F:\experiments\dataset\NYUD", "image-test.lst"), 'r') as f:
    #     test_list = f.readlines()
    test_list = [os.path.split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))
    # 模型
    model = sigma_logit_unetpp.Mymodel(args).cuda()
    # model_chechpoint_path = "F:\\experiments\\dataset\\result\\epoch-0-training-record\\epoch-0-checkpoint.pth"
    model_chechpoint_path = "result-BSDS500/rtv/epoch-24-checkpoint.pth"
    model_chechpoint = torch.load(model_chechpoint_path, map_location='cpu')
    model.load_state_dict(model_chechpoint['state_dict'])
    # 开始测试
    epoch = 25
    output_root = "result-BSDS500/rtv"
    print("----------单尺度测试---------")
    test(model, test_loader, epoch=epoch, test_list=test_list,
         save_dir=os.path.join(output_root, 'epoch-%d-testing-record-SS' % epoch))
    # print("----------多尺度测试----------")
    # multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
    #                 save_dir=os.path.join(output_root, 'epoch-%d-testing-record-MS' % epoch))
    print("----------测试完毕----------")


if __name__ == "__main__":
    main()
