import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from .blocks import *

class MaXDeepLabSBackbone(nn.Module):
    def __init__(
        self,
        layers=[3, 4, 6, 3],
        im_size=640,
        n_heads=8,
        in_channels=3
    ):
        super(MaXDeepLabSBackbone, self).__init__()

        self.base_width = 64
        self.nin = 64
        self.stem = InceptionStem(in_channels, 128)

        self.layer1 = self._make_bottleneck_layer(64, layers[0], stride=1, first_layer=True)
        self.layer2 = self._make_bottleneck_layer(128, layers[1], stride=2)

        kernel_size = im_size // 8
        self.layer3 = self._make_axial_layer(
            256, layers[2], stride=2, n_heads=n_heads, kernel_size=kernel_size
        )

        kernel_size = im_size // 16
        self.layer4 = self._make_axial_layer(
            512, layers[3], stride=1, n_heads=n_heads, kernel_size=kernel_size
        )

    def get_stages(self):
        return [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _make_bottleneck_layer(
        self,
        planes,
        n_blocks,
        stride=1,
        first_layer=False
    ):
        block = Bottleneck
        downsample = None
        first_block_nin = self.nin * 2 if first_layer else self.nin

        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                first_block_nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(first_block_nin, planes, stride, downsample, base_width=self.base_width)
        )
        self.nin = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(
                block(self.nin, planes, base_width=self.base_width)
            )

        return nn.Sequential(*layers)

    def _make_axial_layer(
        self,
        planes,
        n_blocks,
        stride=1,
        n_heads=8,
        kernel_size=56
    ):
        block = AxialBottleneck
        downsample = None
        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                self.nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(self.nin, planes, stride, downsample, self.base_width,
            n_heads=n_heads, kernel_size=kernel_size
            )
        )

        self.nin = planes * block.expansion
        kernel_size = kernel_size // stride
        for _ in range(1, n_blocks):
            layers.append(
                block(self.nin, planes, base_width=self.base_width,
                n_heads=n_heads, kernel_size=kernel_size)
            )

        return nn.Sequential(*layers)

    def forward(self, P):
        P = self.stem(P) #H / 4
        P = self.layer1(P) #H / 4
        P = self.layer2(P) #H / 8
        P = self.layer3(P) #H / 16
        P = self.layer4(P) #H / 16

        return P

class MaXDeepLabSImageNet(nn.Module):
    def __init__(
        self,
        layers=[3, 4, 6, 3],
        im_size=640,
        n_heads=8,
        in_channels=3,
        num_classes=1000
    ):
        super(MaXDeepLabSImageNet, self).__init__()
        self.backbone = MaXDeepLabSBackbone(layers, im_size, n_heads, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
