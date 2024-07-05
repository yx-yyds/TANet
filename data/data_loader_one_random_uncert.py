import json

from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


def resize_img(im):
    h, w = im.shape[-2:]
    new_h, new_w = h + (h % 2), w + (w % 2)
    new_h, new_w = 512, 512
    im = cv2.resize(im, (new_h, new_w))
    return im


def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


# *************************************************
# ************* BIPED(TEED) **************************
# *************************************************
class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 # img_height,
                 # img_width,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 # crop_img=False,
                 # arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        # self.img_height = img_height
        # self.img_width = img_width
        # self.mean_bgr = arg.mean_train if len(arg.mean_train) == 3 else arg.mean_train[:3]
        self.mean_bgr = [103.939, 116.779, 123.68]
        # self.crop_img = crop_img
        # self.arg = arg

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        # file_path = os.path.join(data_root, self.arg.train_list)
        file_path = os.path.join(data_root, "train_pair.lst")
        with open(file_path) as f:
            files = json.load(f)
        for pair in files:
            tmp_img = pair[0]
            tmp_gt = pair[1]
            rtv_weight = os.path.join("rtv", tmp_img[6:])
            sample_indices.append(
                (os.path.join(data_root, tmp_img),
                 os.path.join(data_root, tmp_gt),
                 os.path.join(data_root, rtv_weight)))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path, rtv_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        rtv = cv2.imread(rtv_path, cv2.IMREAD_GRAYSCALE)
        image, label, rtv = self.transform(img=image, gt=label, rtv=rtv)
        return image, label, rtv

    def transform(self, img, gt, rtv):
        # assert img.shapa[:, :, 0] == gt.shape[:, :, 0], f"img形状为{img.shape}， gt形状为{gt.shape}"
        # print(f"img形状为:{img.shape}， gt形状为:{gt.shape}")
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for LDC input and BDCN
        rtv = np.array(rtv, dtype=np.float32)
        if len(rtv.shape) == 3:
            rtv = rtv[:, :, 0]
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        #  400 for BIPEd and 352 for BSDS check with 384
        # crop_size = self.img_height if self.img_height == self.img_width else None  # 448# MDBD=480 BIPED=480/400 BSDS=352
        new_h, new_w = 512, 512

        # BRIND Best for TEDD+BIPED
        img = cv2.resize(img, dsize=(new_h, new_w))
        gt = cv2.resize(gt, dsize=(new_h, new_w))
        # gt[gt > 0.1] += 0.6  # 0.4
        # gt = np.clip(gt, 0., 1.)
        # 我自己的设置
        gt[gt > 0.2] = 1.
        gt[gt <= 0.2] = 0.
        rtv = cv2.resize(rtv, dsize=(new_h, new_w))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        rtv = torch.from_numpy(np.array([rtv])).float()
        return img, gt, rtv


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data="BIPED",
                 # img_height,
                 # img_width,
                 test_list=None,
                 # arg=None
                 ):
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = 'test_pair.lst'
        # self.args = arg
        # self.up_scale = arg.up_scale
        # self.mean_bgr = arg.mean_test if len(arg.mean_test) == 3 else arg.mean_test[:3]
        self.mean_bgr = [104.007, 116.669, 122.679]
        # self.img_height = img_height
        # self.img_width = img_width
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file
            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")
            # list_name = os.path.join(self.data_root, self.test_list)
            list_name = os.path.join(self.data_root, "test.lst")
            if self.test_data.upper() in ['BIPED', 'BRIND', 'UDED', 'ICEDA']:

                # with open(list_name) as f:
                #     files = json.load(f)
                # for pair in files:
                #     tmp_img = pair[0]
                #     tmp_gt = pair[1]
                #     sample_indices.append(
                #         (os.path.join(self.data_root, "imgs/test", tmp_img),
                #          os.path.join(self.data_root, "edges_maps/test", tmp_gt),))
                with open(list_name, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]
                pairs = [line.split() for line in files]

                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    # sample_indices.append(
                    #     (os.path.join(self.data_root, r"edges/imgs/test", tmp_img),
                    #      os.path.join(self.data_root, r"edges/edge_maps/test", tmp_gt),))
                    sample_indices.append((tmp_img, tmp_gt))
            else:
                with open(list_name, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]
                pairs = [line.split() for line in files]

                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, r"imgs\test", tmp_img),
                         os.path.join(self.data_root, r"edge_maps\test", tmp_gt),))
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper() == 'CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == 'BIPED':
            img_dir = os.path.join(self.data_root, 'edges/imgs/test')
            gt_dir = os.path.join(self.data_root, 'edges/edge_maps/test')
        elif self.test_data.upper() == 'CLASSIC':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        # im_shape = [image.shape[0], image.shape[1]]

        image, label = self.transform(img=image, gt=label)

        # return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)
        return image

    def transform(self, img, gt):
        new_h, new_w = 1280, 736
        img_rsize = np.zeros((736, 1280, 3))
        img_rsize[16:, :, :] = img
        img = img_rsize
        # img = cv2.resize(img, dsize=(new_h, new_w))
        # gt[gt< 51] = 0 # test without gt discrimination
        # up scale test image
        # if self.up_scale:
        #     # For TEED BIPBRIlight Upscale
        #     img = cv2.resize(img,(0,0),fx=1.3,fy=1.3)
        #
        # if img.shape[0] < 512 or img.shape[1] < 512:
        #     #TEED BIPED standard proposal if you want speed up the test, comment this block
        #     img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
        # else:
        #     img = cv2.resize(img, (0, 0), fx=1.1, fy=1.1)

        # Make sure images and labels are divisible by 2^4=16
        # if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
        #     img_width = ((img.shape[1] // 8) + 1) * 8
        #     img_height = ((img.shape[0] // 8) + 1) * 8
        #     img = cv2.resize(img, (img_width, img_height))
        #     # gt = cv2.resize(gt, (img_width, img_height))
        # else:
        #     pass
        #     img_width = self.args.test_img_width
        #     img_height = self.args.test_img_height
        #     img = cv2.resize(img, (img_width, img_height))
        #     gt = cv2.resize(gt, (img_width, img_height))
        # # For FPS
        # img = cv2.resize(img, (496,320))

        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR

        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_val_all.lst')

        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]
            label_list = []
            for i_label in range(1, len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root, img_lb_file[i_label]))
                lb = np.asarray(lb['edge_gt'])
                label = torch.from_numpy(lb)
                label = label[1:label.size(0), 1:label.size(1)]
                label = label.float()
                label_list.append(label.unsqueeze(0))
            labels = torch.cat(label_list, 0)
            lb_mean = labels.mean(dim=0).unsqueeze(0)

            lb_std = labels.std(dim=0).unsqueeze(0)
            # cv2.imshow("label", np.asarray(label))
            # cv2.imshow("label_std", np.asarray(labels.mean(dim=0)))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            lb_index = random.randint(2, len(img_lb_file)) - 1
            lb_file = img_lb_file[lb_index]

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":

            lb = scipy.io.loadmat(join(self.root, lb_file))
            lb = np.asarray(lb['edge_gt'])
            label = torch.from_numpy(lb)
            label = label[1:label.size(0), 1:label.size(1)]
            label = label.unsqueeze(0)
            label = label.float()
            file_name = os.path.basename(img_file)
            RTV_weights = cv2.imread(join(self.root, "UAED-BSDS-RTV-weights", splitext(file_name)[0]+".jpg"), 0)
            RTV_weights = np.asarray(RTV_weights)
            RTV_weights = torch.from_numpy(RTV_weights)
            RTV_weights = RTV_weights[1:RTV_weights.size(0), 1:RTV_weights.size(1)]
            RTV_weights = RTV_weights.unsqueeze(0)
            RTV_weights = RTV_weights.float()
            return img, label, RTV_weights

        else:
            return img


def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class NYUD_Loader(data.Dataset):
    """
    Dataloader for NYUDv2
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        """
        There is no threshold for NYUDv2 since it is singlely annotated
        setting should be 'image' or 'hha'
        """
        self.root = root
        self.split = split
        self.threshold = 100
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                self.root, '%s-train.lst' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                self.root, '%s-test.lst' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # scale = 1.0
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            # scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            # if scale < 0.99:  # which means it < 1.0
            #     W = int(scale * pil_image.width)
            #     H = int(scale * pil_image.height)
            #     pil_image = pil_image.resize((W, H))
            # W = (pil_image.width//32 + 1)*32
            # H = (pil_image.height//32 + 1)*32
            pil_image = pil_image.resize((512, 512))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            # x = lb.copy()
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 0
            lb[lb >= threshold] = 1
            # x_lab = np.squeeze(lb)
            # rtv权重
            rtv_weight = Image.open(os.path.join(self.root, 'rtv-weight', img_file))
            rtv_weight = rtv_weight.resize((512, 512))
            rtv = np.asarray(rtv_weight, dtype=np.float32)
            if rtv.ndim == 3:
                rtv = np.squeeze(rtv[:, :, 0])
            assert rtv.ndim == 2
            rtv = rtv[np.newaxis, :, :]

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            if self.split == "test":
                W = (img.width // 32 + 1) * 32
                H = (img.height // 32 + 1) * 32
            else: 
                W = 512
                H = 512
            img = img.resize((W, H))
            img = img.convert('RGB')

        img = self.transform(img)

        if self.split == "train":
            return img, lb, rtv
        else:
            img_name = Path(img_file).stem
            return img

