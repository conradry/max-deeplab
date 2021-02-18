import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['linear_bn_relu', 'conv_bn_relu', 'InceptionStem', 'AxialMultiHeadAttention',
'AxialBottleneck', 'DualPathXF', 'DecoderBottleneck', 'MaskHead']

class linear_bn_relu(nn.Module):
    """
    Default layer for linear operations.
    """
    def __init__(
        self,
        nin,
        nout,
        with_bn=True,
        with_relu=True
    ):
        super(linear_bn_relu, self).__init__()
        self.l1 = nn.Linear(nin, nout, bias=not with_bn)
        self.bn1 = None
        self.relu = None

        if with_bn:
            self.bn1 = nn.BatchNorm1d(nout)
        if with_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #assumed input is (N, B, C)
        out = self.l1(x)
        if self.bn1 is not None:
            #permute to (B, C, N)
            out = out.permute(1, 2, 0)
            out = self.bn1(out)
            out = out.permute(2, 0, 1)
        if self.relu:
            out = self.relu(out)

        return out

class conv_bn_relu(nn.Module):
    """
    Default layer for convolution operations.
    """
    def __init__(
        self,
        nin,
        nout,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        with_bn=True,
        with_relu=True
    ):
        super(conv_bn_relu, self).__init__()
        layers = [
            nn.Conv2d(nin, nout, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=not with_bn)
        ]

        if with_bn:
            layers.append(nn.BatchNorm2d(nout))
        if with_relu:
            layers.append(nn.ReLU(inplace=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class InceptionStem(nn.Module):
    """
    Input stem.
    """
    def __init__(
        self,
        nin=3,
        nout=128
    ):
        super(InceptionStem, self).__init__()
        self.net = nn.Sequential(
            conv_bn_relu(nin, nout // 2, 3, stride=2, padding=1),
            conv_bn_relu(nout // 2, nout // 2, 3, stride=1, padding=1),
            conv_bn_relu(nout // 2, nout, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

    def forward(self, x):
        #(B, NIN, H, W) --> (B, NOUT, H/4, W/4)
        return self.net(x)

class AxialMultiHeadAttention(nn.Module):
    """
    Modified from https://github.com/csrhddlam/axial-deeplab/blob/master/lib/models/axialnet.py.
    """
    def __init__(
        self,
        nin,
        nout,
        n_heads=8,
        kernel_size=40,
        stride=1,
        axis='height'
    ):
        super(AxialMultiHeadAttention, self).__init__()
        self.nin = nin
        self.nout = nout
        self.n_heads = n_heads
        self.head_nin = nout // n_heads
        self.kernel_size = kernel_size
        self.axis = axis

        self.qkv = nn.Sequential(
            nn.Conv1d(nin, nout * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(nout * 2)
        )
        self.bn_attn = nn.BatchNorm2d(n_heads * 3)
        self.bn_output = nn.BatchNorm1d(nout * 2)

        #(HIN * 2, KS * 2 - 1)
        self.pos_emb = nn.Parameter(torch.randn(self.head_nin * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size)[None, :] #(1, KS)
        key_index = torch.arange(kernel_size)[:, None] #(KS, 1)

        #(KS, 1) - (1, KS) --> (KS, KS)
        relative_index = (key_index - query_index) + (kernel_size - 1)
        self.register_buffer('flat_index', relative_index.view(-1)) #(KS * KS)

        self.avg_pool = nn.AvgPool2d(stride, stride=stride) if stride != 1 else None

        self.reset_parameters()

    def reset_parameters(self):
        #initialize qkv conv1d layer
        self.qkv[0].weight.data.normal_(0, math.sqrt(1. / self.nin))
        #and position embedding
        nn.init.normal_(self.pos_emb, 0., math.sqrt(1. / self.head_nin))

    def forward(self, x):
        if self.axis == 'height':
            x = rearrange(x, 'n c h w -> n w c h')
        else:
            x = rearrange(x, 'n c h w -> n h c w')

        N, W, C_in, H = x.shape
        x = rearrange(x, 'n i c j -> (n i) c j')

        #define other useful dimensions
        C_out = self.nout
        kernel_size = self.kernel_size
        n_heads = self.n_heads
        head_nin = self.head_nin
        qkdim = head_nin // 2
        vdim = head_nin
        #NOTE: head_nin * 2 = qkdim + qkdim + vdim

        qkv = self.qkv(x) #(N * W, C_out * 2, H)
        qkv = rearrange(qkv, 'nw (a b) x -> nw a b x', a=n_heads, b=head_nin * 2)
        q, k, v = torch.split(qkv, [qkdim, qkdim, vdim], dim=2)

        embeddings = self.pos_emb[:, self.flat_index]
        embeddings = embeddings.view(head_nin * 2, kernel_size, kernel_size)
        qemb, kemb, vemb = torch.split(embeddings, [qkdim, qkdim, vdim], dim=0)

        #(N * W, n_heads, head_nin / 2, H) x (head_nin / 2, H, H)
        #--> (N * W, n_heads, H, H)
        qr = torch.einsum('bnci,cij->bnij', q, qemb)
        kr = torch.einsum('bnci,cij->bnji', k, kemb) #note the transpose
        qk = torch.einsum('bnci, bncj->bnij', q, k)

        #(N * W, 3 * n_heads, H, H)
        stacked_attn = self.bn_attn(torch.cat([qk, qr, kr], dim=1))
        stacked_attn = rearrange(stacked_attn, 'b (a n) i j -> b a n i j', a=3, n=n_heads)
        stacked_attn = stacked_attn.sum(1) #(N * W, n_heads, H, H)
        attn = F.softmax(stacked_attn, dim=3)

        #attend to values
        sv = torch.einsum('bnij,bncj->bnci', attn, v)
        svemb = torch.einsum('bnij,cij->bnci', attn, vemb)

        #(N * W, n_heads, head_nin, 2 * H) --> (N * W, C_out * 2, H)
        stacked_y = torch.cat([sv, svemb], dim=-1)
        stacked_y = rearrange(stacked_y, 'b n c (k i) -> b (n c k) i', n=n_heads, k=2, i=H)
        y = self.bn_output(stacked_y)

        y = y.view(N, W, C_out, 2, H).sum(dim=-2) #(N, W, C_out, H)

        if self.axis == 'height':
            y = rearrange(y, 'n w c h -> n c h w')
        else:
            y = rearrange(y, 'n h c w -> n c h w')

        if self.avg_pool is not None:
            y = self.avg_pool(y)

        return y

class AxialBottleneck(nn.Module):
    """
    Modified from https://github.com/csrhddlam/axial-deeplab/blob/master/lib/models/axialnet.py.

    ResNet-style bottleneck block with conv3x3 replaced by AxialMultiHeadAttention layers.
    """

    expansion = 4

    def __init__(
        self,
        nin,
        nplanes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        n_heads=8,
        kernel_size=56
    ):
        super(AxialBottleneck, self).__init__()

        width = int(nplanes * (base_width / 64.))
        self.axial_net = nn.Sequential(
            conv_bn_relu(nin, width, kernel_size=1),
            AxialMultiHeadAttention(width, width, n_heads, kernel_size, axis='height'),
            AxialMultiHeadAttention(width, width, n_heads, kernel_size, stride=stride, axis='width'),
            nn.ReLU(inplace=False),
            conv_bn_relu(width, nplanes * self.expansion, kernel_size=1, with_relu=False)
        )

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.axial_net(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DualPathXF(nn.Module):
    """
    The dual path transformer module.
    """

    expansion = 4

    def __init__(
        self,
        nin_pixel,
        nplanes,
        nin_memory,
        stride=1,
        downsample=None,
        base_width=64,
        n_heads=8,
        kernel_size=20
    ):
        super(DualPathXF, self).__init__()

        #nplanes = 1024
        self.p2p = AxialBottleneck(
            nin_pixel, nplanes, stride, downsample, base_width=base_width,
            n_heads=n_heads, kernel_size=kernel_size
        )

        nin_pixel = nplanes * self.expansion

        self.p2m_conv1 = conv_bn_relu(nin_pixel, nplanes, 1)
        self.p2m_qkv = nn.Sequential(
            nn.Conv2d(nplanes, nplanes * 2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(nplanes * 2)
        )
        self.p2m_conv2 = nn.Sequential(
            nn.Conv2d(nplanes, nplanes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(nplanes * self.expansion)
        )

        #memory qkv
        #nplanes = 512
        self.mem_fc1 = linear_bn_relu(nin_memory, nplanes)
        self.mem_qkv = linear_bn_relu(nplanes, nplanes * 2, with_relu=False)
        self.mem_fc2 = linear_bn_relu(nplanes, nin_memory, with_relu=False)
        self.relu = nn.ReLU(inplace=False)

        self.mem_ffn = nn.Sequential(
            linear_bn_relu(nin_memory, nplanes * self.expansion),
            linear_bn_relu(nplanes * self.expansion, nin_memory, with_relu=False)
        )

        #useful dimensions
        self.n_heads = n_heads
        self.head_nin = nplanes // n_heads
        self.dq = self.head_nin // 2
        self.dv = self.head_nin

    def forward(self, x_dict):
        P = x_dict['pixel']
        M = x_dict['memory']

        #useful dimensions
        B, C, H, W = P.size()
        N, B, K = M.size()
        n_heads = self.n_heads #labeled 'i' in einsums
        head_nin = self.head_nin
        dq = self.dq
        dv = self.dv

        #P is pixel (image), M is memory
        P = self.p2p(P)
        P_identity = P
        M_identity = M

        #apply image path qkv
        #(B, C_out * 2, H, W)
        P_qkv = self.p2m_conv1(P)
        P_qkv = self.p2m_qkv(P_qkv)
        P_qkv = rearrange(P_qkv, 'b (i j) h w -> b i j h w', i=n_heads, j=head_nin * 2)
        qp, kp, vp = torch.split(P_qkv, [dq, dq, dv], dim=2)

        #(N, B, K)
        M_qkv = self.mem_fc1(M)
        M_qkv = self.mem_qkv(M_qkv)
        M_qkv = rearrange(M_qkv, 'n b (i j) -> n b i j', i=n_heads, j=head_nin * 2)
        qm, km, vm = torch.split(M_qkv, [dq, dq, dv], dim=3)

        #P2M output it ypa in paper
        #qp: (B, n_heads, dq, h, w), km: (N, B, n_heads, dq)
        p2m = torch.einsum('bijhw,nbij->bnihw', qp, km) #(B, N, n_heads, H, W)
        p2m_attn = F.softmax(p2m, dim=1)
        ypa = torch.einsum('bnihw,nbij->bijhw', p2m_attn, vm) #(B, n_heads, head_nin, H, W)

        #handle m2p and m2m together
        kp = rearrange(kp, 'b i j h w -> b i j (h w)')
        km = rearrange(km, 'n b i j -> b i j n')
        kpm = torch.cat([kp, km], dim=3) #(B, n_heads, dq * 2, H * W + N)

        vp = rearrange(vp, 'b i j h w -> b i j (h w)')
        vm = rearrange(vm, 'n b i j -> b i j n')
        vpm = torch.cat([vp, vm], dim=3) #(B, n_heads, dv * 2, H * W + N)

        m2m_m2p = torch.einsum('nbij,bijl->nbil', qm, kpm) #(N, B, n_heads, H * W + N)
        m2m_m2p_attn = F.softmax(m2m_m2p, dim=-1)
        ymb = torch.einsum('nbil,bijl->nbij', m2m_m2p_attn, vpm) #(N, B, n_heads, head_nin)

        P_out = self.p2m_conv2(rearrange(ypa, 'b i j h w -> b (i j) h w'))
        P_out += P_identity
        P_out = self.relu(P_out)

        M_out = self.mem_fc2(rearrange(ymb, 'n b i j -> n b (i j)'))
        M_out += M_identity
        M_out = self.relu(M_out)

        M_ffn = self.mem_ffn(M_out)
        M_out += M_ffn
        M_out = self.relu(M_out)

        return {'pixel': P_out, 'memory': M_out}

class DecoderBottleneck(nn.Module):
    """
    ResNet-style bottleneck block in the decoder with skip connection
    for encoder layer outputs.
    """

    expansion = 2

    def __init__(
        self,
        nin,
        nplanes,
        upsample_factor=2,
        compression=4
    ):
        super(DecoderBottleneck, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode='bilinear', align_corners=True
        )
        self.relu = nn.ReLU(inplace=False)

        #see Fig 9. of https://arxiv.org/abs/2003.07853
        self.identity_path = nn.Sequential(
            conv_bn_relu(nin, nin // compression, kernel_size=1, with_relu=False),
            self.upsample
        )

        self.bottleneck_path = nn.Sequential(
            conv_bn_relu(nin, nplanes, kernel_size=1),
            self.upsample,
            conv_bn_relu(nplanes, nplanes, kernel_size=3, padding=1),
            conv_bn_relu(nplanes, nin // compression, kernel_size=1, with_relu=False)
        )

        self.encoder_feature_path = nn.Sequential(
            conv_bn_relu(nin // compression, nin // compression, kernel_size=1, with_relu=False)
        )

        self.proj_down = conv_bn_relu(nin // compression, nin // compression, kernel_size=1)

    def forward(self, x, skip):
        identity = self.identity_path(x)
        bottleneck_out = self.bottleneck_path(x)
        skip_out = self.encoder_feature_path(skip)
        out = self.relu(identity + bottleneck_out + skip_out)

        return self.proj_down(out)

class MaskHead(nn.Module):
    """
    Generates the masks prior to multiplication by global memory.
    """
    def __init__(
        self,
        nplanes,
        nout,
        kernel_size=5,
        padding=2,
        separable=True
    ):
        super(MaskHead, self).__init__()
        groups = nplanes if separable else 1
        self.conv5x5 = conv_bn_relu(
            nplanes, nplanes, kernel_size, padding=padding, groups=groups
        )
        self.conv1x1 = conv_bn_relu(nplanes, nout, 1, with_relu=False)

    def forward(self, x):
        return self.conv1x1(self.conv5x5(x))
