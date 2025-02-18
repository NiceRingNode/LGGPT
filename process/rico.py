# -*- coding: utf-8 -*-

import os,json
import numpy as np
from functools import partial

all_labels = [
    'Checkbox', 'Multi-Tab', 'Toolbar', 'Card', 'Pager Indicator', 'Slider', 'Bottom Navigation',
    'Text', 'Video', 'Image', 'Map View', 'Modal', 'Date Picker', 'Advertisement', 'On/Off Switch',
    'Input', 'Web View', 'Drawer', 'Icon', 'Button Bar', 'Background Image', 'List Item', 'Text Button',
    'Number Stepper', 'Radio Button'
]

def is_valid(element,W,H):
    if element['componentLabel'] not in set(all_labels):
        print(element['componentLabel'])
        return False
    x1, y1, x2, y2 = element['bounds']
    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True

def append_child(element,elements):
    if 'children' in element.keys():
        for child in element['children']:
            elements.append(child)
            elements = append_child(child, elements)
    return elements

def annotation_extract(src_root,tgt_root): # http://www.interactionmining.org/rico.html
    cnt = 0
    os.makedirs(tgt_root,exist_ok=True)
    labels = []
    hs,ws = [],[]
    files = [i for i in os.listdir(src_root) if i.endswith('.json')]
    print('file counts:',len(files))
    for i in files:
        file_name = i.split('.')[0]
        with open(f'{src_root}/{i}','r') as f:
            annos = json.load(f)
            W,H = float(annos['bounds'][2]),float(annos['bounds'][3])
            hs.append(H)
            ws.append(W)
            if annos['bounds'][0] != 0 or annos['bounds'][1] != 0 or H < W:
                continue
        elements = append_child(annos,[])
        valid = partial(is_valid,W=W,H=H)
        elements = list(filter(valid,elements))
        num_elements = len(elements)
        if num_elements == 0 or num_elements > 40:
            continue
        class_coords = []
        for e in elements:
            x1,y1,x2,y2 = e['bounds']
            w = x2 - x1
            h = y2 - y1
            class_coords.append([x1,y1,w,h,e['componentLabel']])
            labels.append(e['componentLabel'])
        with open(f'{tgt_root}/{file_name}.txt','w',encoding='utf-8') as f:
            f.write(f'{int(W)},{int(H)}\n')
            lines = []
            for line in class_coords:
                lines.append(f'{line[0]:.2f},{line[1]:.2f},{line[2]:.2f},{line[3]:.2f},{line[4]}')
            f.write('\n'.join(lines))
        cnt += 1
        # break
    print(np.max(hs),np.max(ws))
    print('valid file number:',cnt)
    print('extraction done')

if __name__ == '__main__':
    annotation_extract(src_root='../LGGPT/data-raw/rico_dataset_v0.1_semantic_annotations/semantic_annotations',
                       tgt_root='data-raw/rico-tmp')
