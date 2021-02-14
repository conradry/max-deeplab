import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import rearrange, repeat

#TODO: Optimize code for GPU, consider CUDA version of linear_sum_assignment
#TODO: Validate for multi-GPU training

#@torch.jit.script
def cdice_similarity(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B, N, HW) #probabilities [0, 1]
    target_mask: (B, K, HW) #binary
    """

    input_mask = input_mask.unsqueeze(2) #(B, N, 1, HW)
    target_mask = target_mask.unsqueeze(1) #(B, 1, K, HW)
    #(B, N, 1, HW) * (B, 1, K, HW) --> (B, N, K, HW)
    intersections = torch.sum(input_mask * target_mask, dim=-1)
    cardinalities = torch.sum(input_mask + target_mask, dim=-1)
    dice = ((2. * intersections + eps) / (cardinalities + eps))
    return dice

#@torch.jit.script
def dice_score(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B * K, HW) #probabilities [0, 1]
    target_mask: (B * K, HW) #binary
    """

    dims = tuple(range(1, input_mask.ndimension()))
    intersections = torch.sum(input_mask * target_mask, dims) #(B, N)
    cardinalities = torch.sum(input_mask + target_mask, dims)
    dice = ((2. * intersections + eps) / (cardinalities + eps))
    return dice

class HungarianMatcher(nn.Module):
    """
    Heavily inspired by https://github.com/facebookresearch/detr/blob/master/models/matcher.py.
    """

    def __init__(self):
        super(HungarianMatcher, self).__init__()

    @torch.no_grad()
    def forward(self, input_class_prob, input_mask, target_class, target_mask, target_sizes):
        """
        input_class: (B, N, N_CLASSES) #probabilities
        input mask: (B, N, H, W) #probabilities [0, 1]
        target_class: (B, K) #long indices
        target_mask: (B, K, H, W) #bool
        target_sizes: (B,) #number of masks that are padding (i.e. no class)
        """
        device = input_class_prob.device
        B, N = input_class_prob.size()[:2]
        K = target_class.size(-1)

        #we want similarity matrices to size (B, N, K)
        #where N is number of predicted objects and K is number of gt objects
        #(B, N, C)[(B, N, K)] --> (B, N, K)
        sim_class = input_class_prob.gather(-1, repeat(target_class, 'b k -> b n k', n=N))
        sim_dice = cdice_similarity(input_mask, target_mask)

        #final cost matrix (RQ x SQ from the paper, eqn 9)
        sim = (sim_class * sim_dice).cpu() #(B, N, K)

        #each example in batch, ignore null objects in target (i.e. padding)
        indices = [linear_sum_assignment(s[:, :e], maximize=True) for s,e in zip(sim, target_sizes)]

        #at this junctions everything is matched, now it's just putting
        #the indices into easily usable formats

        input_pos_indices = []
        target_pos_indices = []
        input_neg_indices = []
        input_indices = np.arange(0, N)
        for i, (inp_idx, tgt_idx) in enumerate(indices):
            input_pos_indices.append(torch.as_tensor(inp_idx, dtype=torch.long, device=device))
            target_pos_indices.append(torch.as_tensor(tgt_idx, dtype=torch.long, device=device))
            input_neg_indices.append(
                torch.as_tensor(
                    np.setdiff1d(input_indices, inp_idx), dtype=torch.long, device=device
                )
            )

        #here the lists of indices have variable lengths
        #and sizes; make 1 tensor of size (B * N_pos) for all
        #positives first: shared by input_pos_indices and target_pos_indices
        batch_pos_idx = torch.cat(
            [torch.full_like(pos, i) for i, pos in enumerate(input_pos_indices)]
        )
        batch_neg_idx = torch.cat(
            [torch.full_like(neg, i) for i, neg in enumerate(input_neg_indices)]
        )
        input_pos_indices = torch.cat(input_pos_indices)
        target_pos_indices = torch.cat(target_pos_indices)
        input_neg_indices = torch.cat(input_neg_indices)

        inp_pos_indices = (batch_pos_idx, input_pos_indices)
        tgt_pos_indices = (batch_pos_idx, target_pos_indices)
        inp_neg_indices = (batch_neg_idx, input_neg_indices)
        return inp_pos_indices, tgt_pos_indices, inp_neg_indices

class PQLoss(nn.Module):
    def __init__(self, alpha=0.75, eps=1e-5):
        super(PQLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.xentropy = nn.CrossEntropyLoss(reduction='none')
        self.matcher = HungarianMatcher()
        self.negative_xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, input_class, target_mask, target_class, target_sizes):
        """
        input_class: (B, N, N_CLASSES) #logits
        input mask: (B, N, H, W) #probabilities [0, 1]
        target_class: (B, K) #long indices
        target_mask: (B, K, H, W) #binary
        """
        #apply softmax to get probabilities from logits
        B, N, num_classes = input_class.size()
        input_mask = F.softmax(input_mask, dim=1)
        input_class_prob = F.softmax(input_class, dim=-1)
        input_mask = rearrange(input_mask, 'b n h w -> b n (h w)')
        target_mask = rearrange(target_mask, 'b k h w -> b k (h w)')

        #match input and target
        inp_pos_indices, tgt_pos_indices, neg_indices = self.matcher(
            input_class_prob, input_mask, target_class,
            target_mask, target_sizes
        )

        #select masks and labels by indices
        #(B < len(inp_pos_indices) <= B * K)
        #(0 <= len(neg_indices) <= B * (N - K))
        matched_input_class = input_class[inp_pos_indices]
        matched_input_class_prob = input_class_prob[inp_pos_indices]
        matched_target_class = target_class[tgt_pos_indices]
        negative_class = input_class[neg_indices]

        matched_input_mask = input_mask[inp_pos_indices]
        matched_target_mask = target_mask[tgt_pos_indices]
        negative_mask = input_mask[neg_indices]

        #NP is len(inp_pos_indices)
        #NN is len(neg_indices)
        with torch.no_grad():
            class_weight = matched_input_class_prob.gather(-1, matched_target_class[:, None]) #(NP,)
            dice_weight = dice_score(matched_input_mask, matched_target_mask, self.eps) #(NP,)

        cross_entropy = self.xentropy(matched_input_class, matched_target_class) #(NP,)
        dice = dice_score(matched_input_mask, matched_target_mask, self.eps) #(NP,)
    
        #eqn 10
        #(1 - dice) is so that the minimum loss value is 0 and not -1
        l_pos = (class_weight * (1 - dice) + dice_weight * cross_entropy).mean()

        #eqn 11
        negative_target_class = torch.zeros(
            size=(len(negative_class),), dtype=target_class.dtype, device=target_class.device
        )
        l_neg = self.negative_xentropy(negative_class, negative_target_class).mean()
        
        #eqn 12
        return self.alpha * l_pos + (1 - self.alpha) * l_neg

#-----------------------
### Auxiliary Losses ###
#-----------------------

class InstanceDiscLoss(nn.Module):
    def __init__(self, temp=0.3, eps=1e-5):
        super(InstanceDiscLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    def forward(self, mask_features, target_mask, target_sizes):
        """
        mask_features: (B, D, H, W) #g
        target_mask: (B, K, H, W) #m
        """

        #downsample input and target by 4 to get (B, H/4, W/4)
        mask_features = mask_features[..., ::4, ::4]
        target_mask = target_mask[..., ::4, ::4]

        device = mask_features.device

        #eqn 16
        t = torch.einsum('bdhw,bkhw->bkd', mask_features, target_mask)
        t = F.normalize(t, dim=-1) #(B, K, D)

        #get batch and mask indices from target_sizes
        batch_indices = []
        mask_indices = []
        for bi, size in enumerate(target_sizes):
            mindices = torch.arange(0, size, dtype=torch.long, device=device)
            mask_indices.append(mindices)
            batch_indices.append(torch.full_like(mindices, bi))

        batch_indices = torch.cat(batch_indices, dim=0) #shape: (torch.prod(target_sizes), )
        mask_indices = torch.cat(mask_indices, dim=0)

        #create logits and apply temperature
        logits = torch.einsum('bdhw,bkd->bkhw', mask_features, t)
        logits = logits[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)
        logits /= self.temp

        #select target_masks
        m = target_mask[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)

        #flip so that there are HW examples for torch.prod(target_sizes) classes
        logits = rearrange(logits, 'k h w -> (h w) k')
        m = rearrange(m, 'k h w -> (h w) k')

        #eqn 17
        numerator = torch.logsumexp(m * logits, dim=-1) #(HW,)
        denominator = torch.logsumexp(logits, dim=-1) #(HW,)
        
        #log of quotient is difference of logs
        return (-numerator + denominator).mean()

class MaskIDLoss(nn.Module):
    def __init__(self):
        super(MaskIDLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, N, H, W) #logits
        target_mask: (B, H, W) #long indices of maskID in N
        """
        return self.xentropy(input_mask, target_mask)

class SemanticSegmentationLoss(nn.Module):
    def __init__(self, method='cross_entropy'):
        super(SemanticSegmentationLoss, self).__init__()
        if method != 'cross_entropy':
            raise NotImplementedError
        else:
            #they don't specify the loss function
            #could be regular cross entropy or
            #dice loss or focal loss etc.
            #keep it simple for now
            self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, NUM_CLASSES, H, W) #logits
        target_mask: (B, H, W) #long indices
        """
        return self.xentropy(input_mask, target_mask)

class MaXDeepLabLoss(nn.Module):
    def __init__(
        self,
        pq_loss_weight=3,
        instance_loss_weight=1,
        maskid_loss_weight=0.3,
        semantic_loss_weight=1,
        alpha=0.75,
        temp=0.3,
        eps=1e-5
    ):
        super(MaXDeepLabLoss, self).__init__()
        self.pqw = pq_loss_weight
        self.idw = instance_loss_weight
        self.miw = maskid_loss_weight
        self.ssw = semantic_loss_weight
        self.pq_loss = PQLoss(alpha, eps)
        self.instance_loss = InstanceDiscLoss(temp, eps)
        self.maskid_loss = MaskIDLoss()
        self.semantic_loss = SemanticSegmentationLoss()

    def forward(self, input_tuple, target_tuple):
        """
        input_tuple: (input_masks, input_classes, input_semantic_segmentation) Tensors
        target_tuple: (gt_masks, gt_classes, gt_semantic_segmentation) NestedTensors
        """
        input_masks, input_classes, input_ss = input_tuple
        gt_masks, gt_classes, gt_ss = [t.tensors for t in target_tuple]
        target_sizes = target_tuple[0].sizes

        pq = self.pq_loss(input_masks, input_classes, gt_masks, gt_classes, target_sizes)
        instdisc = self.instance_loss(input_masks, gt_masks.float(), target_sizes)

        #create the mask for maskid loss using argmax on ground truth
        maskid = self.maskid_loss(input_masks, gt_masks.argmax(1))
        semantic = self.semantic_loss(input_ss, gt_ss)

        loss_items = {'pq': pq.item(), 'semantic': semantic.item(), 
                      'maskid': maskid.item(), 'instdisc': instdisc.item()}

        total_loss = self.pqw * pq + self.ssw * semantic + self.miw * maskid + self.idw * instdisc 

        return total_loss, loss_items
