# -*- coding:utf-8 -*-

import os
import numpy as np

def get_column_num(annos,img_w): # annos已经是split之后的了，坐标转成了float
    annos = np.asarray(annos)
    left = annos[:,0]
    right = left + annos[:,2]
    mid = int(0.5 * img_w)
    mask1 = left > mid
    mask2 = right < mid
    if np.sum(mask1) > 0 or np.sum(mask2) > 0:
        return 2
    else:
        return 1

def generate_prompts_pure_value(anno_roots=['./data-raw/publaynet/train','./data-raw/publaynet/val']):
    if not isinstance(anno_roots,list):
        anno_roots = [anno_roots]
    for root in anno_roots:
        print(root)
        tgt_root = root.replace('-tmp','')
        os.makedirs(tgt_root,exist_ok=True)
        files = os.listdir(root)
        for file in files:
            file_name = file.split('.')[0]
            with open(f'{root}/{file}','r',encoding='utf-8') as f:
                annos = f.readlines()
            annos = [l.strip() for l in annos]
            img_w,img_h = list(map(lambda x:int(float((x))),annos[0].split(',')))
            annos = annos[1:]
            annos = [l.split(',') for l in annos]
            annos = np.array(annos)
            class_names = np.array(annos[:,-1])
            annos = [list(map(lambda x:float(x),l[:-1])) for l in annos]
            column = get_column_num(annos,img_w)
            prompt = f'{len(annos)};'
            coord_text = ''
            for i in range(len(annos)):
                coord_text += f'{class_names[i].lower()},'
                coord_text += f'{annos[i][0]},'
                coord_text += f'{annos[i][1]},'
                coord_text += f'{annos[i][2]},'
                coord_text += f'{annos[i][3]};'
            prompt += coord_text
            prompt += f'{img_w},{img_h};{column}'
            # prompt = prompt[:-1] # 最后的分号不要
            # print(prompt)
            with open(f'{tgt_root}/{file_name}.txt','w',encoding='utf-8') as f:
                f.write(prompt)

def transform2onefile(anno_roots=['./data/publaynet/train','./data/publaynet/val']):
    if not isinstance(anno_roots,list):
        anno_roots = [anno_roots]
    for root in anno_roots:
        all_annos = []
        for file in os.listdir(root):
            with open(f'{root}/{file}','r',encoding='utf-8') as f:
                annos = f.readlines()[0].strip()
            all_annos.append(f'{file} {annos}')
        os.makedirs(os.path.dirname(root.replace('data-raw','data')),exist_ok=True)
        tgt_path = root.replace('data-raw','data') + '.txt'
        with open(tgt_path,'w',encoding='utf-8') as f:
            f.write('\n'.join(all_annos))

def split_rico(src_root='./data/rico40',tgt_root='./data/rico'):
    os.makedirs(tgt_root,exist_ok=True)
    files = os.listdir(src_root)
    train_files = np.random.choice(files,size=int(len(files) * 0.9),replace=False)
    test_files = list(set(files) - set(train_files))
    all_annos = []
    for file in train_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file}+{annos}')
    with open(f'{tgt_root}/train.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))
    all_annos = []
    for file in test_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file}+{annos}')
    with open(f'{tgt_root}/test.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))

def split_magazine(src_root='./data/magazine-files',tgt_root='./data/magazine'):
    os.makedirs(tgt_root,exist_ok=True)
    files = os.listdir(src_root)
    train_files = np.random.choice(files,size=int(len(files) * 0.9),replace=False)
    test_files = list(set(files) - set(train_files))
    all_annos = []
    for file in train_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file} {annos}')
    with open(f'{tgt_root}/train.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))
    all_annos = []
    for file in test_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file} {annos}')
    with open(f'{tgt_root}/test.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))

def split_slide(src_root='./data/ppt-files',tgt_root='./data/ppt'):
    os.makedirs(tgt_root,exist_ok=True)
    files = os.listdir(src_root)
    train_files = np.random.choice(files,size=int(len(files) * 0.9),replace=False)
    test_files = list(set(files) - set(train_files))
    all_annos = []
    for file in train_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file} {annos};slide')
        # all_annos.append(f'{file} {annos}')
    with open(f'{tgt_root}/train.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))
    all_annos = []
    for file in test_files:
        with open(f'{src_root}/{file}','r',encoding='utf-8') as f:
            annos = f.readlines()[0].strip()
        all_annos.append(f'{file} {annos};slide')
        # all_annos.append(f'{file} {annos}')
    with open(f'{tgt_root}/test.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_annos))

def merge_dataset_balanced():
    rico_files = {}
    with open(f'../data/rico/train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    for l in lines:
        fn,prompt = l.split('+')
        rico_files[fn] = prompt.lower()
    magazine_files = {}
    with open(f'../data/magazine/train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    for l in lines:
        fn,prompt = l.split(' ')
        magazine_files[fn] = prompt.lower()
    publaynet_files = {}
    with open(f'../data/publaynet/train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    for l in lines:
        fn,prompt = l.split(' ')
        publaynet_files[fn] = prompt.lower()
    slide_files = {}
    with open(f'../data/slide/train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        slide_files[l.split()[0]] = l.split()[1]
 
    tgt_root = '../data/unified'
    os.makedirs(tgt_root,exist_ok=True)
    pub_train = list(publaynet_files.keys())
    rico_train = list(rico_files.keys()) * 7
    mag_train = list(magazine_files.keys()) * 95
    slide_train = list(slide_files.keys()) * 111

    train_annos = []
    for i in pub_train:
        annos = publaynet_files[i]
        annos = annos + ';article'
        train_annos.append(f'{i} {annos}')
    for i in rico_train:
        annos = rico_files[i]
        annos = annos + ';App-UI'
        train_annos.append(f'{i} {annos}')
    for i in mag_train:
        annos = magazine_files[i]
        annos = annos + ';magazine'
        train_annos.append(f'{i} {annos}')
    for i in slide_train:
        annos = slide_files[i]
        train_annos.append(f'{i} {annos}')
    print(len(train_annos),len(pub_train),len(rico_train),len(mag_train),len(slide_train))
    with open(f'{tgt_root}/train.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(train_annos))

def transform_test_data():
    with open('../data/publaynet/val.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    lines = [l.lower().replace('+',' ') for l in lines]
    lines = [l + ';article' for l in lines]
    with open('../data/publaynet/val_prompt.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(lines))

    with open('../data/rico/test.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    lines = [l.lower().replace('+',' ') for l in lines]
    lines = [l + ';App-UI' for l in lines]
    with open('../data/rico/test_prompt.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(lines))

    with open('../data/magazine/test.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    lines = [l.lower().replace('+',' ') for l in lines]
    lines = [l + ';magazine' for l in lines]
    with open('../data/magazine/test_prompt.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(lines))
            
if __name__ == '__main__':
    generate_prompts_pure_value(['../data-raw/publaynet-tmp/train','../data-raw/publaynet-tmp/val'])
    generate_prompts_pure_value('../data-raw/rico-tmp')
    generate_prompts_pure_value('../data-raw/magazine-tmp')
    generate_prompts_pure_value('../data-raw/slide-tmp')

    transform2onefile(['../data-raw/publaynet/train','../data-raw/publaynet/val'])
    split_rico('../data-raw/rico','../data/rico')
    split_magazine('../data-raw/magazine','../data/magazine')
    split_slide('../data-raw/slide','../data/slide')

    merge_dataset_balanced()
    transform_test_data()