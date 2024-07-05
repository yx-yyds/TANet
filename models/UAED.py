import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params
import numpy as np
from PIL import Image
from torchvision import transforms


class Attention(nn.Module):  # 注意力什么也不干？
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):  # 就是使用双线性插值实现两倍上采样？
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            _, _, H, W = skip.shape
            x = F.interpolate(x, size=(H, W), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        
        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        sides = []
        # ------我加的side输出
        for i in range(self.depth+1):
            sides.insert(0, dense_x[f"x_{0}_{i}"])
        # return dense_x[f"x_{0}_{self.depth}"]
        return sides


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return
        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0]: self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1]: self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)

            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    encoders = {
        "efficientnet-b7": {
            "encoder": EfficientNetEncoder,
            "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
            "params": {
                "out_channels": (3, 64, 48, 80, 224, 640),
                "stage_idxs": (11, 18, 38, 55),
                "model_name": "efficientnet-b7",
                # "depth": 5
            },
        },
    }
    Encoder = encoders[name]["encoder"]  # efficientNet模型(EfficientNetEncoder)

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))  # 加载预训练权重

    encoder.set_in_channels(in_channels, pretrained=weights is not None)

    return encoder


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Mymodel(nn.Module):
    def __init__(self, args, encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=3, classes=1):
        super(Mymodel, self).__init__()

        self.encoder_depth = 5
        
        self.decoder_use_batchnorm = True,
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.decoder_attention_type = None

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=self.encoder_depth,
            weights=encoder_weights,
        )
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )
        side_modules = []
        for i in range(self.encoder_depth):
            side_modules.append(
                SegmentationHead(
                    in_channels=self.decoder_channels[self.encoder_depth-1-i],
                    out_channels=classes,
                    kernel_size=3,
                )
            )
        self.side_modules = nn.ModuleList(side_modules)
        self.out_head = nn.Conv2d(in_channels=len(side_modules), out_channels=classes, kernel_size=1)
        self.decoder_1 = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )
        self.args = args

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        
        features = self.encoder(x)
        # decoder_output = self.decoder(*features)
        decoder_outputs = self.decoder(*features)
        # results = self.segmentation_head(decoder_output)
        side_outputs = []
        for i in range(len(decoder_outputs)):
            side_output = self.side_modules[i](decoder_outputs[i])
            side_output = F.interpolate(side_output, size=(img_H, img_W), mode="bilinear", align_corners=False)
            side_outputs.append(side_output)
        # center crop
        results = self.out_head(torch.cat(side_outputs, dim=1))
        results = crop(results, img_H, img_W, 0, 0)
        if self.args.distribution == "beta":
            results = nn.Softplus()(results)
        # -------------方差-----------------
        # decoder_output_1 = self.decoder_1(*features)
        # results_1 = self.segmentation_head_1(decoder_output_1)
        # # center crop
        # std = crop(results_1, img_H, img_W, 0, 0)
        # if self.args.distribution != "residual":
        #     std = nn.Softplus()(std)

        return results


# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="123")
    parser.add_argument("--distribution",
                        type=str,
                        default='gs',
                        help="the output distribution")
    args = parser.parse_args()
    batch_size = 4
    image_width = 352
    image_height = 352
    device = 'cuda' if torch.cuda.is_available() else 'gpu'
    images = torch.rand(batch_size, 3, image_height, image_width).to(device)
    # image = Image.open("./data/my_image/2018.jpg")
    # image = transforms.ToTensor()(image)
    # image = image.float()
    model = Mymodel(args=args).to(device)
    # model.load_state_dict(torch.load("F:\\experiments\\UAED-main\\checkpoints\\pretrain_epoch-19-checkpoint_VOC.pth",
    #                                  map_location='cuda'))
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter("logs")
    # writer.add_graph(model, images)
    # writer.close()
    output = model(images)


