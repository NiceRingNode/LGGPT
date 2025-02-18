# -*- coding:utf-8 --

import json,os
from functools import partial
import numpy as np

# original image counts: 335703
# valid image counts: 333848
# original image counts: 11245
# valid image counts: 11208

all_labels = {'text','title','list','table','figure'}

def is_valid(class_coords,W,H):
    if class_coords[-2] not in set(all_labels):
        print(class_coords[-2])
        return False
    x1,y1,w,h = class_coords[:4]
    x2 = x1 + w
    y2 = y1 + h
    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True

def annotation_extract(src_path,tgt_root):
    mode = src_path.split('/')[-1].split('.')[0]
    with open(src_path,'r',encoding='utf-8') as f:
        info = json.loads(f.read())
    category_dict = {each['id']:each['name'] for each in info['categories']}
    id_name_dict = {each['id']:each['file_name'] for each in info['images']}
    image_coords = {}
    image_hw = {each['file_name']:[each['width'],each['height']] for each in info['images']}
    for anno in info['annotations']:
        img_id = anno['image_id']
        img_name = id_name_dict[img_id]
        if image_coords.get(img_name) is None:
            image_coords[img_name] = []
        bbox = anno['bbox'] # x,y,w,h
        bbox.append(category_dict[anno['category_id']])
        bbox.append(anno['area'])
        image_coords[img_name].append(bbox)
    print('original image counts:',len(image_coords))
    os.makedirs(f'{tgt_root}/{mode}',exist_ok=True) # mode是train或val
    cnt = 0
    for k in image_coords:
        img_name = k.split('.jpg')[0]
        W,H = image_hw[k]
        valid = partial(is_valid,W=W,H=H)
        class_coords = image_coords[k]
        class_coords = list(filter(valid,class_coords))
        if len(class_coords) == 0 or len(class_coords) > 25:
            continue
        with open(f'{tgt_root}/{mode}/{img_name}.txt','w',encoding='utf-8') as f:
            f.write(f'{image_hw[k][0]},{image_hw[k][1]}\n')
            lines = []
            for line in class_coords:
                lines.append(f'{line[0]:.2f},{line[1]:.2f},{line[2]:.2f},{line[3]:.2f},{line[5]},{line[4]}')
            f.write('\n'.join(lines))
        cnt += 1
    print('valid image counts:',cnt)
    print('extraction done')
    
def filter_outliers(root='data-raw/publaynet/train'): # 修正了两张图，PMC5553901_00002和PMC3279813_00002，前面是改了到373，后面是直接删了那个框
    for p in os.listdir(root):
        with open(f'{root}/{p}','r',encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[1:]
            lines = [l.strip() for l in lines]
            lines = list(map(lambda x:x.split(','),lines))
        lines = np.array(lines)[:,:-2]
        lines = np.array([list(map(lambda x:float(x),l)) for l in lines])
        # print(lines[lines < 0],lines[lines > 1024])
        if len(lines[lines < 0]) > 0:
            print(0,f'{root}/{p}')
        if len(lines[lines > 1024]) > 0:
            print(1024,f'{root}/{p}')

if __name__ == '__main__':
    annotation_extract('../LGGPT/data-raw/publaynet-image/train.json','data-raw/publaynet-tmp')
    annotation_extract('../LGGPT/data-raw/publaynet-image/val.json','data-raw/publaynet-tmp')
    # filter_outliers('data-raw/publaynet/train')
    # filter_outliers('data-raw/publaynet/val')