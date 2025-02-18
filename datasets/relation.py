# -*- coding:utf-8 -*-
import sys
sys.path.append('..')
import random
from enum import Enum
from itertools import combinations,product
from utils import convert_xywh_to_ltrb

class SizeRelation():
    unknown = 'size_unk'
    smaller = 'smaller'
    equal = 'equal'
    larger = 'larger'

class LocRelation():
    unknown = 'loc_unk'
    left = 'left'
    top = 'top'
    right = 'right'
    bottom = 'bottom'
    overlap = 'overlap'

def detect_size_relation(b1,b2):
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    # print('area',a1,a2)
    alpha = 0.1
    if (1 - alpha) * a1 < a2 < (1 + alpha) * a1:
        return SizeRelation.equal
    elif a1 < a2:
        return SizeRelation.smaller
    else:
        return SizeRelation.larger

def detect_loc_relation(box1,box2):
    l1, t1, r1, b1 = convert_xywh_to_ltrb(box1)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box2)
    # print('box1',l1,t1,r1,b1)
    # print('box2',l2,t2,r2,b2)
    # if b1 >= t2:
    #     return LocRelation.top
    # elif b2 >= t1: # 
    #     return LocRelation.bottom
    if b2 <= t1: # y轴向下就是这样的
        return LocRelation.bottom
    elif b1 <= t2:
        return LocRelation.top
    elif r2 <= l1: # 对的
        return LocRelation.right
    elif r1 <= l2:
        return LocRelation.left
    else:
        return LocRelation.overlap

class RelationGenerator:
    def __init__(self,seed=None,edge_ratio=0.1,use_v1=False):
        self.edge_ratio = edge_ratio
        self.use_v1 = use_v1
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)
        self.size_relation_unk = SizeRelation.unknown
        self.loc_relation_unk = LocRelation.unknown

    def __call__(self,bboxes): # bboxes是量化之前的
        n = len(bboxes)
        relations_all = list(product(range(2),combinations(range(n),2)))
        size = min(int(len(relations_all) * self.edge_ratio),2) # 最多3
        relation_samples = set(self.generator.sample(relations_all,size))
        # relation_samples = {(0, (1, 3)), (1, (1, 3))}
        size_indices,loc_indices,size_attr,loc_attr = [],[],[],[] # edge_indices是(src,tgt)，attr是大小和方向
        for i,j in combinations(range(n),2):
            bi,bj = bboxes[i],bboxes[j]
            if self.use_v1:
                if (0,(i,j)) in relation_samples:
                    relation_size = detect_size_relation(bi,bj)
                    relation_loc = detect_loc_relation(bi,bj)
                else:
                    relation_size = SizeRelation.unknown
                    relation_loc = LocRelation.unknown
            else:
                if (0,(i,j)) in relation_samples:
                    relation_size = detect_size_relation(bi,bj)
                else:
                    relation_size = SizeRelation.unknown

                if (1,(i,j)) in relation_samples: # 0和1应该分别代表size和loc的relation
                    relation_loc = detect_loc_relation(bi,bj)
                else:
                    relation_loc = LocRelation.unknown
            if relation_loc != self.loc_relation_unk:
                loc_indices.append((i,j))
                loc_attr.append(relation_loc)
            if relation_size != self.size_relation_unk:
                size_indices.append((i,j))
                size_attr.append(relation_size)
            # print(i,j,'rel 三剑客',relation,relation_size,relation_loc)
        # print(edge_indices,edge_attr)
        return loc_indices,size_indices,loc_attr,size_attr
    
if __name__ == '__main__':
    print(SizeRelation)
    l = [[37.59,78.29,251.09,135.41],
        [37.59,245.65,251.09,271.41],
        [37.59,549.0,251.06,83.08],
        [37.59,632.69,251.08,83.14],
        [306.6,58.09,25.41,9.0]
    ]
    m = RelationGenerator()
    k = m(l)
    print(k)