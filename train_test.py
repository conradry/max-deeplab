from max_deeplab.model import MaXDeepLabS
from max_deeplab.losses import MaXDeepLabLoss
from datasets.coco_panoptic import build
from torch.optim import Adam

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

model = MaXDeepLabS(im_size=640, n_classes=202, n_masks=128).cuda()
criterion = MaXDeepLabLoss().cuda()
optimizer = Adam(model.parameters(), lr=3e-4)

P = batch['image'].cuda()
M = torch.randn((128, 8, 256)).cuda()

masks = batch['masks'].cuda()
labels = batch['labels'].cuda()
semantic = batch['semantic_mask'].cuda()
target_tuple = (masks, labels, semantic)

for i in range(50):

    optimizer.zero_grad()

    out = model(P, M)
    loss, loss_items = criterion(out, target_tuple)
    loss.backward()
    optimizer.step()

    print(f'epoch {i}: ', loss_items)
