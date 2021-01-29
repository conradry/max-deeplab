# MaX-DeepLab

Unofficial implementation of MaX-DeepLab for Instance Segmentation: https://arxiv.org/abs/2012.00759v1.

<figure>
  <img height=300 src="./architecture.png"></img>
</figure>

## Current Status

This repository is under active development. Currently, only the MaX-DeepLab-S architecture is putatively complete. At the moment this code is best used as a reference. The ultimate goal is to develop and release a pre-trained model that reproduces the topline results of the paper.

- [x] Axial Attention block
- [x] Dual Path Transformer block
- [x] MaX-DeepLab-S architecture
- [ ] PQ-style loss
- [ ] Hungarian Matcher
- [ ] Sine Position Embedding (or something better)
- [ ] Auxiliary losses (Instance discrimination, Mask-ID cross-entropy, Semantic Segmentation)
- [ ] Optimize model by replacing einsums
- [ ] Encoder pre-training on ImageNet
- [ ] MaX-DeepLab-S training on COCO Panoptic
- [ ] MaX-DeepLab-L???

MaX-DeepLab has a complex architecture, training procedure and loss function; and luckily the paper is short on technical details. Any suggestions or help are appreciated.

## Usage

```python
from max_deeplab.model import MaXDeepLabS

model = MaXDeepLabS(im_size=64, n_classes=80)

P = torch.randn((4, 3, 64, 64))
M = torch.randn((50, 4, 256))

mask_out, classes = model(P, M)
print(mask_out.shape, classes.shape)
>>> (torch.Size([4, 50, 64, 64]), torch.Size([4, 50, 80]))

num_params = []
for pn, p in model.named_parameters():
    num_params.append(np.prod(p.size()))

print(f'{sum(num_params):,} total parameters.')
>>> 61,849,316 total parameters.

```

(Reported number of parameters in the paper is 61.9M)
