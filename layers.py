import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from einops import rearrange

class conv_bn_relu(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    ):
        super(conv_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                nin, nout, kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False
            ),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class InceptionStem(nn.Module):
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
            nn.MaxPool2d(3, stride=2)
        )

    def forward(self, x):
        #(B, NIN, H, W) --> (B, NOUT, H/4, W/4)
        return self.net(x)

class AxialMHA(nn.Module):
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
        super(AxialMHA, self).__init__()
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

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(nplanes * (base_width / 64.))

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_bn_relu(nin, width, kernel_size=1)

        self.axial_attn = nn.Sequential(
            AxialMHA(width, width, n_heads, kernel_size, axis='height'),
            AxialMHA(width, width, n_heads, kernel_size, stride=stride, axis='height'),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(width, nplanes * self.expansion, 1, bias=False),
            nn.BatchNorm2d(nplanes * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.axial_attn
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MemoryQKV(nn.Module):
    """
    Just standard SelfAttention
    """
    def __init__(
        nplanes,
        n_heads=8,
        attn_p=0,
        resid_p=0
    ):
        super(M2MAttention, self).__init__()
        assert nplanes % n_heads == 0
        self.n_heads = n_heads

        self.mem_qkv = nn.Linear(nplanes, nplanes * 2)
        #batchnorm or not?

    def forward(self, x):
        return self.mem_qkv(x)

class DualPathXF(nn.Module):
    expansion = 1
    def __init__(
        self,
        nin_pixel,
        nin_memory,
        n_heads=n_heads,
        kernel_size=20,
    ):
        super(DualPathXF).__init__()
        self.p2p = AxialBottleneck(nin, nout, n_heads, kernel_size=kernel_size)

        #I'm assuming that attention is multihead
        #it's never stated explicitly...

        #pixel qkv
        #maybe another conv before?
        self.p2m_conv1 = conv_bn_relu(nin_pixel, nin_pixel, 1)
        self.p2m_qkv = nn.Sequential(
            nn.Conv2d(nin_pixel, nin_pixel * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(nin_pixel * 2)
        )
        self.p2m_conv2 = nn.Sequential(
            nn.Conv2d(nin_pixel, nin_pixel, kernel_size=1, bias=False),
            nn.BatchNorm2d(nin_pixel)
        )

        #memory qkv
        self.mem_fc1 = nn.Sequential(
            nn.Linear(nin_memory, nin_pixel),
            nn.ReLU(inplace=True) #is there a relu, what about normalization?
        )
        self.mem_qkv = nn.Linear(nin_pixel, nin_pixel * 2) #normalization layer?
        self.mem_fc2 = nn.Linear(nin_pixel, nin_memory) #normalization layer?
        self.relu = nn.ReLU(inplace=True)

        self.mem_ffn = nn.Sequential(
            nn.Linear(nin_memory, nin_pixel),
            nn.ReLU(inplace=True), #again, normalization layer?
            nn.Linear(nin_pixel, nin_memory)
        )

        #useful dimensions
        self.head_nin = nout // n_heads
        self.dq = self.head_nin // 2
        self.dv = self.head_nin

        def forward(self, P, M):
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
            kpm = torch.cat([kp, km], dim=2) #(B, n_heads, dq * 2, H * W + N)

            vp = rearrange(vp, 'b i j h w -> b i j (h w)')
            vm = rearrange(vm, 'n b i j -> b i j n')
            vpm = torch.cat([vp, vm], dim=2) #(B, n_heads, dv * 2, H * W + N)

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

            return P_out, M_out

def MaXDeepLabSEncoder(nn.Module):
    def __init__(
        self,
        blocks=[Bottleneck, Bottleneck, AxialBottleneck, DualPathXF],
        layers=[3, 4, 6, 3],
        im_size=640
    ):
        super(MaXDeepLabSEncoder, self).__init__()

        self.base_width = 64
        self.nin = 128
        self.stem = InceptionStem(3, self.nin)

        self.layer1 = self._make_layer(blocks[0], 64, layers[0])

    def _make_layer(
        self,
        block,
        planes,
        n_blocks,
        stride=1,
        **kwargs
    ):
        downsample = None
        if stride != 1 or self.nin != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.nin, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(nin_pixel)
            )

        layers = []
        layers.append(block(self.nin, planes, stride, downsample,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
