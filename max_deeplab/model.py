import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models.resnet import Bottleneck
from .blocks import *

class MaXDeepLabSEncoder(nn.Module):
    def __init__(
        self,
        layers=[3, 4, 6, 3],
        im_size=640,
        nin_memory=256,
        n_heads=8
    ):
        super(MaXDeepLabSEncoder, self).__init__()

        self.base_width = 64
        self.nin = 64
        self.nin_memory = nin_memory
        self.stem = InceptionStem(3, 128)

        self.layer1 = self._make_bottleneck_layer(64, layers[0], stride=1, first_layer=True)
        self.layer2 = self._make_bottleneck_layer(128, layers[1], stride=2)

        kernel_size = im_size // 8
        self.layer3 = self._make_axial_layer(
            256, layers[2], stride=2, n_heads=n_heads, kernel_size=kernel_size
        )

        kernel_size = im_size // 16
        self.layer4 = self._make_dualpath_layer(
            512, layers[3], n_heads=n_heads, kernel_size=kernel_size
        )

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

    def _make_dualpath_layer(
        self,
        planes,
        n_blocks,
        stride=1,
        base_width=64,
        n_heads=8,
        kernel_size=20
    ):
        block = DualPathXF
        downsample = None
        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                self.nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(self.nin, planes, self.nin_memory, stride, downsample,
            base_width=base_width, n_heads=n_heads, kernel_size=kernel_size)
        )

        self.nin = planes * block.expansion
        kernel_size = kernel_size // stride
        for _ in range(1, n_blocks):
            layers.append(
                block(self.nin, planes, self.nin_memory, stride,
                base_width=base_width, n_heads=n_heads, kernel_size=kernel_size)
            )

        return nn.Sequential(*layers)

    def forward(self, P, M):
        P1 = self.stem(P) #H / 4
        P2 = self.layer1(P1) #H / 4
        P3 = self.layer2(P2) #H / 8
        P4 = self.layer3(P3) #H / 16
        DP = self.layer4({'pixel': P4, 'memory': M}) #H / 16
        P5, M = DP['pixel'], DP['memory']

        return [P1, P2, P3, P4, P5], M

class MaXDeepLabSDecoder(nn.Module):
    def __init__(
        self,
        nin_pixel=2048,
        nplanes=512,
        nin_memory=256,
        im_size=640,
        n_heads=8,
        n_classes=19
    ):
        super(MaXDeepLabSDecoder, self).__init__()
        self.dual_path = DualPathXF(
            nin_pixel, nplanes, nin_memory,
            base_width=64, n_heads=8, kernel_size=im_size // 16
        )

        self.bottleneck1 = DecoderBottleneck(
            nin_pixel, nplanes, upsample_factor=2, compression=4
        )

        nin_pixel = nin_pixel // 4
        nplanes = nplanes // 4
        self.bottleneck2 = DecoderBottleneck(
            nin_pixel, nplanes, upsample_factor=2, compression=2
        )

        nin_pixel = nin_pixel // 2
        self.mask_head = MaskHead(nin_pixel)

        self.mem_mask = linear_bn_relu(nin_memory, nin_pixel, with_relu=False)
        self.mem_class = nn.Linear(nin_memory, n_classes)

    def forward(self, P_features, M):
        P1, P2, P3, P4, P5 = P_features
        dp_out = self.dual_path({'pixel': P5, 'memory': M})
        P_up, M = dp_out['pixel'], dp_out['memory']

        P_up = self.bottleneck1(P_up, P3)
        P_up = self.bottleneck2(P_up, P2)
        mask_up = self.mask_head(P_up) #(B, D, H/4, W/4)

        #handle memory multiplication
        mem_mask = self.mem_mask(M) #(N, B, D)
        mask_out = torch.einsum('nbd,bdhw->bnhw', mem_mask, mask_up)

        classes = self.mem_class(M) #(N, B, n_classes)
        classes = rearrange(classes, 'n b c -> b n c')

        return mask_out, classes

class MaXDeepLabS(nn.Module):
    def __init__(
        self,
        im_size=640,
        n_heads=8,
        n_classes=80
    ):
        super(MaXDeepLabS, self).__init__()
        self.encoder = MaXDeepLabSEncoder(im_size=im_size, n_heads=n_heads)
        self.decoder = MaXDeepLabSDecoder(im_size=im_size, n_heads=n_heads, n_classes=n_classes)

    def forward(self, P, M):
        """
        P: pixel tensor (B, 3, H, W)
        M: memory tensor (N, B, 256)
        """

        fmaps, mem = self.encoder(P, M)
        mask_out, classes = self.decoder(fmaps, mem)
        return mask_out, classes