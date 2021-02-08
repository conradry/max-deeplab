"""
Copied with modification from:
https://github.com/facebookresearch/detr/blob/master/datasets/coco_panoptic.py
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json, os
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from panopticapi.utils import rgb2id
from PIL import Image

class CocoPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms

    def __len__(self):
        return len(self.coco['images'])

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = os.path.join(self.img_folder, ann_info['file_name'].replace('.png', '.jpg'))
        ann_path = os.path.join(self.ann_folder, ann_info['file_name'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = [mask.astype(np.uint8) for mask in masks]
            labels = [ann['category_id'] for ann in ann_info['segments_info']]

        output = {}
        output['image'] = img
        output['masks'] = masks

        if self.transforms is not None:
            output = self.transforms(**output)

        output['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        output['labels'] = torch.tensor(labels, dtype=torch.long)
        output['masks'] = torch.from_numpy(np.stack(output['masks'], axis=0))

        #create semantic segmentation mask
        output['semantic_mask'] = torch.einsum('nhw,n->hw', output['masks'].long(), output['labels'])

        return output

def make_coco_transforms(image_set='train', image_size=(640, 640)):
    normalize = A.Sequential([
        A.Normalize(), #imagenet norms default
        ToTensorV2()
    ])

    if image_set == 'train':
        transforms = A.Compose([
            A.PadIfNeeded(*image_size, border_mode=0), #pad with zeros
            A.RandomResizedCrop(*image_size),
            A.HorizontalFlip(),
            normalize
        ])

    elif image_set == 'val':
        transforms = A.Compose([
            A.Resize(*image_size),
            normalize
        ])

    else:
        raise ValueError(f'{image_set} not recognized!')

    return transforms

def build(image_set, config):
    img_folder_root = config['coco_path']
    ann_folder_root = os.path.join(img_folder_root, 'annotations')
    assert os.path.exists(img_folder_root), f'provided COCO path {img_folder_root} does not exist'
    assert os.path.exists(ann_folder_root), f'provided COCO path {ann_folder_root} does not exist'

    PATHS = {
        "train": ("train2017", 'panoptic_train2017.json'),
        "val": ("val2017", 'panoptic_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = os.path.join(img_folder_root, img_folder)
    ann_folder = os.path.join(ann_folder_root, f'panoptic_{img_folder}')
    ann_file = os.path.join(ann_folder_root, ann_file)

    dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_coco_transforms(image_set, config['image_size']))

    return dataset
