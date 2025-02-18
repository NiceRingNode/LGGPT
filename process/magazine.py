# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET

labels = ['text','image','headline','text-over-image','headline-over-image']

def annotation_extract(src_root,tgt_root):
    os.makedirs(tgt_root,exist_ok=True)
    files = [i for i in os.listdir(src_root) if i.endswith('.xml')]
    print('file counts:',len(files))
    hs,ws = [],[]
    for i in files:
        root = ET.parse(f'{src_root}/{i}').getroot()
        W = float(root.find('size/width').text)
        H = float(root.find('size/height').text)
        file_name = root.find('filename').text
        elements = root.findall('layout/element')
        if len(elements) == 0 or len(elements) > 25:
            continue
        class_coords = []
        for e in elements:
            px = list(map(float,e.get('polygon_x').split()))
            py = list(map(float,e.get('polygon_y').split()))
            x1,x2 = min(px),max(px)
            y1,y2 = min(py),max(py)
            w = x2 - x1
            h = y2 - y1
            hs.append(h)
            ws.append(w)
            class_coords.append([x1,y1,w,h,e.get('label')])

        with open(f'{tgt_root}/{file_name}.txt','w',encoding='utf-8') as f:
            f.write(f'{int(W)},{int(H)}\n')
            lines = []
            for line in class_coords:
                lines.append(f'{line[0]:.2f},{line[1]:.2f},{line[2]:.2f},{line[3]:.2f},{line[4]}')
            f.write('\n'.join(lines))
        # break
    # print(np.max(hs),np.max(ws))
    print('extraction done')

if __name__ == '__main__':
    annotation_extract(src_root='../data-raw/MagLayout/layoutdata/annotations',
                       tgt_root='../data-raw/magazine-tmp')