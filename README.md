# MaX-DeepLab

Unofficial implementation of MaX-DeepLab for Instance Segmentation: https://arxiv.org/abs/2012.00759v1.

<figure>
  <img height=300 src="./architecture.png"></img>
</figure>

## Status

Only the MaX-DeepLab-S architecture is putatively implemented. Primarily, this code is intended as a reference; I can't make any guarantees that it will reproduce the results of the paper.

- [x] Axial Attention block
- [x] Dual Path Transformer block
- [x] MaX-DeepLab-S architecture
- [x] Hungarian Matcher
- [x] PQ-style loss
- [x] Auxiliary losses (Instance discrimination, Mask-ID cross-entropy, Semantic Segmentation)
- [x] Coco Panoptic Dataset
- [x] Simple inference

## Usage

See example.ipynb.
