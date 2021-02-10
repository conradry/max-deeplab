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
- [x] Hungarian Matcher
- [x] PQ-style loss
- [x] Auxiliary losses (Instance discrimination, Mask-ID cross-entropy, Semantic Segmentation)
- [ ] Optimize model runtime
- [ ] Encoder pre-training on ImageNet
- [ ] MaX-DeepLab-S training on COCO Panoptic
- [ ] MaX-DeepLab-L???

MaX-DeepLab has a complex architecture, training procedure and loss function. Any suggestions or help are appreciated.

## Usage

```python
from max_deeplab.model import MaXDeepLabS
from max_deeplab.losses import MaXDeepLabLoss
from datasets.coco_panoptic import build
from util.misc import collate_fn

config = {}
config['image_size'] = (640, 640)
config['coco_path'] = '../../datasets/coco_panoptic/' #directory with image and annotation data
data = build('train', config)

#create a dataloader that collates batch with padding
#see utils.misc.collate_fn
padding_dict = {'image': 0, 'masks': 0, 'semantic_mask': 0, 'labels': 201, 'image_id': 0} #201 is 'no_class' class
sizes_dict = {'image': None, 'masks': 128, 'semantic_mask': None, 'labels': 128, 'image_id': None}
collate_lambda = lambda b: collate_fn(b, padding_dict, sizes_dict)
loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate_lambda)

batch = iter(loader).next()

#returns a dictionary of NestedTensors (each has a 'tensors' and 'sizes' attribute)
#'sizes' is the number of ground truth masks for an image that are not from padding
print(batch['image'].tensors.size(), batch['masks'].tensors.size(),
batch['labels'].tensors.size(), batch['semantic_mask'].tensors.size())
>>> (torch.Size([8, 3, 640, 640]), torch.Size([8, 128, 640, 640]), torch.Size([8, 128]), torch.Size([8, 640, 640]))

model = MaXDeepLabS(im_size=640, n_classes=202, n_masks=128)
criterion = MaXDeepLabLoss()

num_params = []
for pn, p in model.named_parameters():
    num_params.append(np.prod(p.size()))
print(f'{sum(num_params):,} total parameters.')
>>> 61,873,172 total parameters.

P = batch['image']
M = torch.randn((128, 8, 256))

mask_out, classes, semantic = model(P, M)
print(mask_out.shape, classes.shape, semantic.shape)
>>> (torch.Size([8, 128, 640, 640]), torch.Size([8, 128, 202], torch.Size([8, 202, 640, 640])))

loss = criterion((mask_out, classes, semantic), (batch['masks'], batch['labels'], batch['semantic_mask']))
print(loss) #returns loss value and a dict of loss items for each loss
>>> (tensor(-4.5043),
 {'pq': 0.0689038634300232,
  'instdisc': -11.75303840637207,
  'maskid': 5.25504732131958,
  'semantic': 5.465500354766846})
```

(Note: Reported number of parameters in the paper is 61.9M)
