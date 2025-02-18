# -*- coding: utf-8 -*-

import torch,os,time,multiprocessing
from torch import nn
from itertools import chain
import numpy as np
from fid import FIDNetV3,FIDNetV3Slide
from pytorch_fid.fid_score import calculate_frechet_distance
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from utils import convert_xcycwh_to_ltrb

class FIDMultiModel:
    def __init__(self,config,device):
        for k in config:
            if k == 'slide':
                setattr(self,f'model_{k}',FIDNetV3Slide(num_label=config[k]['num_classes'],max_bbox=config[k]['num_positions']).to(device))
            else:
                setattr(self,f'model_{k}',FIDNetV3(num_label=config[k]['num_classes'],max_bbox=config[k]['num_positions']).to(device))
            state_dict = torch.load(config[k]['fid_weight_path'],map_location=device)
            state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
            cur_model = getattr(self,f'model_{k}')
            cur_model.load_state_dict(state_dict)
            cur_model.requires_grad_(False)
            cur_model.eval()
        # self.model = FIDNetV3(num_label=num_classes,max_bbox=num_positions).to(device)    
        self.gt_feats = []
        self.gen_feats = []
        self.data_type_dict = {'article':'publaynet','App-UI':'rico','magazine':'magazine','slide':'ppt'}

    @torch.no_grad()    
    def extract_features(self,bboxes,classes,padding_mask,gt=False,data_type='article'):
        # padding_mask: 'False' for valid points, 'True' for padding points, 我们这里直接设全1，因为只有一个样本
        cur_model = getattr(self,f'model_{self.data_type_dict[data_type]}')
        feats = cur_model.extract_features(bboxes,classes,padding_mask).cpu().numpy()
        features = self.gt_feats if gt else self.gen_feats
        features.append(feats)

    @property
    def features(self):
        return self.gt_feats,self.gen_feats
    
    def load_features(self,gt,gen):
        self.gt_feats = gt
        self.gen_feats = gen

    def compute_fid_score(self):
        gt_feats = np.concatenate(self.gt_feats)
        gen_feats = np.concatenate(self.gen_feats)
        mu1 = np.mean(gt_feats,axis=0)
        sigma1 = np.cov(gt_feats,rowvar=False)
        mu2 = np.mean(gen_feats,axis=0)
        sigma2 = np.cov(gen_feats,rowvar=False)
        return calculate_frechet_distance(mu1,sigma1,mu2,sigma2)

def compute_alignment(bbox,mask):
    """
    Computes some alignment metrics that are different to each other in previous works.
    Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """
    S = bbox.size(1)

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xcycwh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log(1 - X)

    # original
    # return X.sum(-1) / mask.float().sum(-1)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0

    # Y = rearrange(Y.abs(), "b x s1 s2 -> b s1 x s2")
    # Y = reduce(Y, "b x s1 s2 -> b x", "min")
    # Y = rearrange(Y.abs(), " -> b s1 x s2")
    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    results = {
        "alignment-ACLayoutGAN": score,
        "alignment-LayoutGAN++": score_normalized,
        "alignment-NDN": score_Y,
    }
    return results

def compute_overlap(bbox,mask): # 这里的bbox，都是xc,yc,w,h，而不是左上角
    """
    Based on
    (i) Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    (ii) LAYOUTGAN: GENERATING GRAPHIC LAYOUTS WITH WIREFRAME DISCRIMINATORS (ICLR2019)
    https://arxiv.org/abs/1901.06767
    "percentage of total overlapping area among any two bounding boxes inside the whole page."
    At least BLT authors seems to sum. (in the MSCOCO case, it surpasses 1.0)
    """
    B, S = mask.size()
    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xcycwh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xcycwh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    # diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    # ai = ai.masked_fill(diag_mask, 0)
    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=ai.device)
    batch_mask[:, idx, idx] = True
    ai = ai.masked_fill(batch_mask, 0)

    ar = torch.nan_to_num(ai / a1)  # (B, S, S)

    # original
    # return ar.sum(dim=(1, 2)) / mask.float().sum(-1)

    # fixed to avoid the case with single bbox
    score = reduce(ar, "b s1 s2 -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    ids = torch.arange(S)
    ii, jj = torch.meshgrid(ids, ids, indexing="ij")
    ai[repeat(ii >= jj, "s1 s2 -> b s1 s2", b=B)] = 0.0
    overlap = reduce(ai, "b s1 s2 -> b", reduction="sum")

    results = {
        "overlap-ACLayoutGAN": score,
        "overlap-LayoutGAN++": score_normalized,
        "overlap-LayoutGAN": overlap,
    }
    return results

def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        # note: maximize is supported only when scipy >= 1.4
        # print(iou)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N

def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]

def compute_iou(box_1,box_2,generalized=False):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, nn.FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xcycwh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xcycwh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    # print(l1,t1,r1,b1)
    # print(l2,t2,r2,b2)
    # print(a1,a2,ai)
    # print()
    iou = ai / (au + 1e-8) # 有可能生成出来的只是一条线，这时候IOU计算出来就是nan，实际是置0就好

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out

def compute_maximum_iou(
    layouts_1,
    layouts_2,
    disable_parallel=True,
    n_jobs=None,
):
    """
    Computes Maximum IoU [Kikuchi+, ACMMM'21]
    """
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    # print(c2bl_1)
    # print(c2bl_2)
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    # to check actual number of layouts for evaluation
    # ans = 0
    # for x in args:
    #     ans += len(x[0])
    if disable_parallel:
        scores = [__compute_maximum_iou(a) for a in args]
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()
    
def __compute_bbox_sim(
    bboxes_1: np.ndarray,
    category_1: np.int64,
    bboxes_2: np.ndarray,
    category_2: np.int64,
    C_S: float = 2.0,
    C: float = 0.5,
) -> float:
    # bboxes from diffrent categories never match
    if category_1 != category_2:
        return 0.0

    cx1, cy1, w1, h1 = bboxes_1
    cx2, cy2, w2, h2 = bboxes_2

    delta_c = np.sqrt(np.power(cx1 - cx2, 2) + np.power(cy1 - cy2, 2))
    delta_s = np.abs(w1 - w2) + np.abs(h1 - h2)
    area = np.minimum(w1 * h1, w2 * h2)
    alpha = np.power(np.clip(area, 0.0, None), C)

    weight = alpha * np.power(2.0, -1.0 * delta_c - C_S * delta_s)
    return weight

def __compute_docsim_between_two_layouts(layouts_1_layouts_2,max_diff_thresh=3):
    layouts_1, layouts_2 = layouts_1_layouts_2
    bboxes_1, categories_1 = layouts_1
    bboxes_2, categories_2 = layouts_2

    N, M = len(bboxes_1), len(bboxes_2)
    if N >= M + max_diff_thresh or N <= M - max_diff_thresh:
        return 0.0

    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([__compute_bbox_sim(bboxes_1[i], categories_1[i], bboxes_2[j], categories_2[j]) for i, j in zip(ii, jj)]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)

    if len(scores[ii, jj]) == 0:
        # sometimes, predicted bboxes are somehow filtered.
        return 0.0
    else:
        return scores[ii, jj].mean()

def compute_docsim(layouts_gt,layouts_generated,disable_parallel=True,n_jobs=None,):
    """
    Compute layout-to-layout similarity and average over layout pairs.
    Note that this is different from layouts-to-layouts similarity.
    """
    args = list(zip(layouts_gt, layouts_generated))
    if disable_parallel:
        scores = []
        for arg in args:
            scores.append(__compute_docsim_between_two_layouts(arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_docsim_between_two_layouts, args)
    return np.array(scores).mean()