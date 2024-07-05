from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        efficient_v2 = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.encoder = efficient_v2.features
        self.out_stage = [0, 2, 3, 5, 7]
        self.out_channels = [3, 24, 48, 80, 176, 512]

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i in self.out_stage:
                outputs.append(x)
        return outputs


class VGGBlock(nn.Module):  # 两个卷积层
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        self.encoder = Encoder()
        nb_filter = [24, 48, 80, 176, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        _, x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(input)
        # x0_0 = self.conv0_0(input)
        # x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.up(self.final1(x0_1))
            output2 = self.up(self.final2(x0_2))
            output3 = self.up(self.final3(x0_3))
            output4 = self.up(self.final4(x0_4))
            return [output1, output2, output3, output4]

        else:
            output = self.up(self.final(x0_4))
            return output


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'gpu'
    images = torch.rand(10, 3, 512, 512).to(device)
    model = NestedUNet().to(device)
    outputs = model(images)
    print("模型参数为：{}MB".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024))
