import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from .blocks import *
from .backbone import MaXDeepLabSBackbone

class MaXDeepLabSEncoder(nn.Module):
    def __init__(
        self,
        layers=[3, 4, 6, 3],
        im_size=640,
        nin_memory=256,
        n_heads=8,
        in_channels=3
    ):
        super(MaXDeepLabSEncoder, self).__init__()

        self.base_width = 64
        self.nin = 64
        self.nin_memory = nin_memory
        backbone = MaXDeepLabSBackbone(layers, im_size, n_heads, in_channels)
        stages = backbone.get_stages()
        self.stem = stages[0]
        self.layer1 = stages[1]
        self.layer2 = stages[2]
        self.layer3 = stages[3]

        #overwrite layer 4 and replace with dual path
        del stages[4]
        kernel_size = im_size // 16
        self.nin *= 16
        self.layer4 = self._make_dualpath_layer(
            512, layers[3], n_heads=n_heads, kernel_size=kernel_size
        )

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
        n_classes=19,
        n_masks=50
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
        self.mask_head = MaskHead(nin_pixel, n_masks)

        self.mem_mask = nn.Sequential(
            linear_bn_relu(nin_memory, nin_memory),
            linear_bn_relu(nin_memory, n_masks, with_relu=False)
        )

        self.mem_class = nn.Sequential(
            linear_bn_relu(nin_memory, nin_memory),
            nn.Linear(nin_memory, n_classes)
        )

        self.fg_bn = nn.BatchNorm2d(n_masks)
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True
        )


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
        mask_out = self.fg_bn(mask_out)
        mask_out = self.upsample(mask_out)

        classes = self.mem_class(M) #(N, B, n_classes)
        classes = rearrange(classes, 'n b c -> b n c')

        return mask_out, classes

class MaXDeepLabS(nn.Module):
    def __init__(
        self,
        im_size=640,
        n_heads=8,
        n_classes=80,
        n_masks=50
    ):
        super(MaXDeepLabS, self).__init__()
        self.encoder = MaXDeepLabSEncoder(im_size=im_size, n_heads=n_heads)
        self.decoder = MaXDeepLabSDecoder(
            im_size=im_size, n_heads=n_heads,
            n_classes=n_classes, n_masks=n_masks
        )

        self.semantic_head = nn.Sequential(
            conv_bn_relu(2048, 256, 5, padding=2, groups=256),
            conv_bn_relu(256, n_classes, 1, with_bn=False, with_relu=False),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )

        self.global_memory = nn.Parameter(torch.randn((n_masks, 256)), requires_grad=True)

    def forward(self, x):
        return self.conv1x1(self.conv5x5(x))

    def forward(self, P):
        """
        P: pixel NestedTensor (B, 3, H, W)
        """

        #P is a nested tensor, extract the image data
        #see utils.misc.NestedTensor
        P, sizes = P.decompose()
        M = repeat(self.global_memory, 'n k -> n b k', b=P.size(0))

        fmaps, mem = self.encoder(P, M)
        semantic_mask = self.semantic_head(fmaps[-1])
        mask_out, classes = self.decoder(fmaps, mem)
        return mask_out, classes, semantic_mask
