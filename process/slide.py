# -*- coding: utf-8 -*-

import os,hdf5storage,scipy
import numpy as np

def spase(src_path='labels',tgt_path='slide'): # https://cvhci.anthropomatik.kit.edu/~mhaurile/spase/
    os.makedirs(tgt_path,exist_ok=True)
    files = os.listdir(src_path)
    print('spase original files:',len(files))
    cnt = 0
    for file in files:
        data = scipy.io.loadmat(f'{src_path}/{file}')
        if 'labelNames' not in data.keys() or 'imageSize' not in data.keys() or 'labels' not in data.keys(): continue
        # print(f'{src_path}/{file}')
        label_names = data['labelNames']
        h,w = data['imageSize'][0][:2]
        class_coords = []
        for i,c in enumerate(range(data['labels'].shape[-1])):
            clas = label_names[i][0][0]
            cur_bbox = data['labels'][:,:,c]
            y,x = np.where(cur_bbox != 0)
            if len(y) == 0 or len(x) == 0: continue
            left = min(x)
            top = min(y)
            iw = max(x) - min(x)
            ih = max(y) - min(y)
            class_coords.append([left,top,iw,ih,clas])
        if len(class_coords) == 0: continue
        with open(f'{tgt_path}/{file.split(".")[0]}.txt','w',encoding='utf-8') as f:
            f.write(f'{w},{h}\n')
            lines = []
            for line in class_coords:
                lines.append(f'{line[0]:.2f},{line[1]:.2f},{line[2]:.2f},{line[3]:.2f},{line[4]}')
            f.write('\n'.join(lines))
        cnt += 1
    print('SPaSe all valid files:',cnt)

def wise(src_path='labels',tgt_path='slide'): # https://cvhci.anthropomatik.kit.edu/~mhaurile/wise/
    os.makedirs(tgt_path,exist_ok=True) # 1328 files
    files = os.listdir(src_path)
    print('wise original files:',len(files))
    cnt = 0
    for file in files:
        # if 'copy' in file: continue
        data = hdf5storage.loadmat(f'{src_path}/{file}')
        if 'labelNames' not in data.keys() or 'imageSize' not in data.keys() or 'labels' not in data.keys(): continue
        # print(f'{src_path}/{file}')
        label_names = data['labelNames']
        h,w,c = data['imageSize'][0]
        class_coords = []
        for i,c in enumerate(range(data['labels'].shape[-1])):
            clas = label_names[i][0][0][0]
            if clas in ['unprocessed','unlabeled']: continue
            cur_bbox = data['labels'][:,:,c]
            y,x = np.where(cur_bbox != 0)
            if len(y) == 0 or len(x) == 0: continue
            left = min(x)
            top = min(y)
            iw = max(x) - min(x)
            ih = max(y) - min(y)
            class_coords.append([left,top,iw,ih,clas])
        if len(class_coords) == 0: continue
        with open(f'{tgt_path}/{file.split(".")[0]}.txt','w',encoding='utf-8') as f:
            f.write(f'{w},{h}\n')
            lines = []
            for line in class_coords:
                lines.append(f'{line[0]:.2f},{line[1]:.2f},{line[2]:.2f},{line[3]:.2f},{line[4]}')
            f.write('\n'.join(lines))
        cnt += 1
    print('WiSe all valid files:',cnt)

if __name__ == '__main__':
    wise(src_path='../LGGPT/data-raw/WiSe/labels',tgt_path='data-raw/slide-tmp')
    spase(src_path='../LGGPT/data-raw/SPaSe/labels',tgt_path='data-raw/slide-tmp')