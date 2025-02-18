# -*- coding:utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
import re
from copy import deepcopy
import torch
from transformers import GPT2Tokenizer,AutoTokenizer
from .relation import RelationGenerator

class Layout(Dataset):
    def __init__(self,data_path,tokenizer,tokenizer_config,train=True,condition='C'):
        super().__init__()
        self.publaynet_classes = ['text','title','list','table','figure']
        rico_classes = [
            "Text","Image","Icon","Text Button","List Item","Input","Background Image","Card","Web View",
            "Radio Button","Drawer","Checkbox","Advertisement","Modal","Pager Indicator","Slider",
            "On/Off Switch","Button Bar","Toolbar","Number Stepper","Multi-Tab","Date Picker","Map View",
            "Video","Bottom Navigation"
        ]
        self.rico_classes = ['-'.join([j.lower() for j in i.split(' ')]) for i in rico_classes]
        self.magazine_classes = ['text','image','headline','text-over-image','headline-over-image']
        slide_classes = [
            'Affiliation','Comments','Date','Diagramm','Enumeration','Footnote',
            'Functions','Heading','ImageCaption','Legend','Logos','Maps',
            'OtherImg','Paragraph','PresTitle','Pseudocode','Realistic','SlideNr',
            'Syn/Drawings','Tables','TitleSlide','Website','hwMathExpr','typedMathExpr',
        ]
        self.slide_classes = [i.lower() for i in slide_classes]
        self.classes = {
            'article': self.publaynet_classes,
            'App-UI': self.rico_classes,
            'magazine': self.magazine_classes,
            'slide': self.slide_classes,
        }
        self.classes2idx = {
            'article': {c:i for i,c in enumerate(self.publaynet_classes)},
            'App-UI': {c:i for i,c in enumerate(self.rico_classes)},
            'magazine': {c:i for i,c in enumerate(self.magazine_classes)},
            'slide': {c:i for i,c in enumerate(self.slide_classes)},
        }
        self.scale_max_len = 1024
        self.App_UI_prompt_template = [
            'Generate a layout of {}.',
            'Generate a layout of {}, with {} elements.',
            'Generate a layout of {}, with {} elements and {} columns.',
            'Design a highly flexible UI interface for a multi-functional application.',
            'Design an intuitive UI interface for a broad user base.',
            'Show me a dynamic and diverse UI interface design.',
        ]
        self.magazine_prompt_template = [
            'Generate a layout of {}.',
            'Generate a layout of {}, with {} elements.',
            'Generate a layout of {}, with {} elements and {} columns.',
            'Please create a versatile magazine layout.',
            'I need an informative magazine cover.',
            'Design a flexible layout for a magazine publisher.',
        ]
        self.article_prompt_template = [
            'Generate a layout of {}.',
            'Generate a layout of {}, with {} elements.',
            'Generate a layout of {}, with {} elements and {} columns.',
            'I need an article layout with various presentation options.',
            'Create a clean and organized article layout for a scientific journal article.',
            'Design a professional article layout for a journal.',
        ]
        self.slide_prompt_template = [
            'Generate a layout of {}.',
            'Generate a layout of {}, with {} elements.',
            'Generate a layout of {}, with {} elements and {} columns.',
            'I want a slide with diverse presentation options.',
            'Design an eye-catching slide for a conference presentation.',
            'Please generate a slide for content targeted at a wide audience.',
        ]

        with open(data_path,'r',encoding='utf-8') as f:
            labels = f.readlines()
            labels = [l.strip().split(' ') for l in labels]
            # labels = labels[8000:8392]
        self.names = np.asarray(labels)[:,0].tolist()
        self.labels = np.asarray(labels)[:,1].tolist()
        self.train = train
        print(tokenizer_config,tokenizer)
        if tokenizer == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_config)
        elif tokenizer in ['tinyllamav1.1','nsfw']:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config,add_bos_token=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config,add_bos_token=False,add_eos_token=False)
        print(self.tokenizer.unk_token,self.tokenizer.bos_token,self.tokenizer.eos_token)
        if tokenizer == 'tinyllamav1.1':
            self.pad_token = self.tokenizer.unk_token
            self.bos_token = self.tokenizer.bos_token
            self.eos_token = self.tokenizer.eos_token
        elif tokenizer == 'nsfw':
            self.pad_token = self.tokenizer.unk_token
            self.bos_token = self.tokenizer.unk_token
            self.eos_token = self.tokenizer.unk_token
        else:
            self.pad_token = self.tokenizer.eos_token
            self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token != None else self.pad_token
            self.eos_token = self.tokenizer.eos_token if self.tokenizer.bos_token != None else self.pad_token
            assert self.pad_token == self.bos_token == self.eos_token
        self.bbox_sep_token = ';'
        self.input_sep_token = '#'
        self.refine_token = 'refine'
        self.norefine_token = 'unrefine'
        self.input_sep_token_id = self.tokenizer(self.input_sep_token,return_tensors='pt')['input_ids'].squeeze().numpy()
        self.bbox_sep_token_id = self.tokenizer(self.bbox_sep_token,return_tensors='pt')['input_ids'].squeeze().numpy()
        
        self.tokenizer_name = tokenizer
        self.condition = condition
        self.relation_generator = RelationGenerator(seed=123)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self):
        return self.tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        if self.train:
            return self.encode_train(index)
        else:
            return self.encode_test(index,self.condition)
       
    def add_perturbations_relative(self,class_bboxes,W,H,p=0.5):
        coords = [int(W),int(H),int(W),int(H)]
        class_bboxes_copy = deepcopy(class_bboxes)
        refine_flags = np.zeros_like(class_bboxes_copy)
        num_bboxes = len(class_bboxes)
        for i in range(num_bboxes):
            for j in range(1,5):
                if np.random.random() < p:
                    noise = torch.normal(0.,0.01,(1,)).numpy()[0]
                    perturb = int(round(coords[j - 1] * noise,0))
                    class_bboxes_copy[i][j] = class_bboxes_copy[i][j] + perturb
                    # if j in [1,2]: # x和y
                    #     class_bboxes_copy[i][j] = np.clip(class_bboxes_copy[i][j],0,coords[j - 1])
                    # else: # w和h
                    #     pos = class_bboxes_copy[i][j - 2] # 拿这个时候的x或y
                    #     class_bboxes_copy[i][j] = np.clip(class_bboxes_copy[i][j],0,coords[j - 1] - 1 - pos)

                    if j in [1,2]: # x和y
                        class_bboxes_copy[i][j] = np.clip(class_bboxes_copy[i][j],(j - 1) * self.scale_max_len,coords[j - 1] + (j - 1) * self.scale_max_len - 1)
                    else: # w和h
                        pos = class_bboxes_copy[i][j - 2] - (j - 3) * self.scale_max_len # 拿这个时候的x或y
                        class_bboxes_copy[i][j] = np.clip(class_bboxes_copy[i][j],(j - 1) * self.scale_max_len,
                                                          coords[j - 1] + (j - 1) * self.scale_max_len - 1 - pos)
                    if perturb != 0:
                        refine_flags[i][j] = 1
        return class_bboxes_copy,refine_flags

    def generate_relation(self,bboxes,types):
        # print('in rel',len(bboxes))
        loc_indices,size_indices,loc_attr,size_attr = self.relation_generator(bboxes)
        # print(loc_indices,size_indices,loc_attr,size_attr)
        relation_tokens = []
        for (i,j),attr in zip(loc_indices,loc_attr):
            s = f'{types[i]} {i} {attr} {types[j]} {j}'
            relation_tokens.append(s)
        for (i,j),attr in zip(size_indices,size_attr):
            s = f'{types[i]} {i} {attr} {types[j]} {j}'
            relation_tokens.append(s)
        relation_tokens = ','.join(relation_tokens) if len(relation_tokens) > 0 else ''
        # print(relation_tokens)
        return relation_tokens

    def generate_natural_prompt(self,layout_type,num_elements,num_columns):
        prompt_candidates = getattr(self,f'{"_".join(layout_type.split("-"))}_prompt_template')
        if layout_type == 'App-UI':
            layout_type = 'App UI'
        idx = np.random.choice(len(prompt_candidates),size=1)[0]
        if idx == 0:
            prompt = [self.norefine_token,prompt_candidates[idx].format(layout_type)]
        elif idx == 1:
            prompt = [self.norefine_token,prompt_candidates[idx].format(layout_type,num_elements)]
        elif idx == 2:
            prompt = [self.norefine_token,prompt_candidates[idx].format(layout_type,num_elements,num_columns)]
        else:
            prompt = [self.norefine_token,prompt_candidates[idx]]
        return prompt

    def scale_coords(self,class_coord_tokens,W,H):
        # WHs = [int(W),int(H),int(W),int(H)]
        max_side = max(W,H)
        W_scaled = int(W / max_side * self.scale_max_len)
        H_scaled = int(H / max_side * self.scale_max_len)
        class_coord_scaled = deepcopy(class_coord_tokens)
        for i in range(len(class_coord_tokens)):
            for j in range(1,5):
                normed_value = class_coord_tokens[i][j] / max_side # 归一化了，根据长边归一化，letter box
                scaled_value = normed_value * self.scale_max_len
                scaled_value = int(round(scaled_value,0))
                class_coord_scaled[i][j] = scaled_value
        # print(class_coord_tokens,W,H)
        # print(class_coord_scaled,W_scaled,H_scaled)
        return class_coord_scaled,W_scaled,H_scaled
    
    def coordinates_encoding(self,class_bboxes):
        # print(class_bboxes)
        class_bboxes_copy = deepcopy(class_bboxes)
        for i in range(len(class_bboxes_copy)):
            for j in range(2,len(class_bboxes_copy[i])):
                class_bboxes_copy[i][j] += self.scale_max_len * (j - 1)
        return class_bboxes_copy
    
    def encode_train(self,index):
        label_str = self.labels[index]
        label_list = label_str.split(';') # label_list[0] can not be 0
        W,H = [float(l) for l in label_list[-3].split(',')]
        column = int(label_list[-2])
        layout_type = label_list[-1]
        label_list = label_list[:-3]
        
        cond = np.random.choice(['U','R','O'],size=1,p=[0.15,0.1,0.75]) # O: Overall; R: Refinement-related, U: unknown
        if cond == 'O':
            prompt = [self.norefine_token,layout_type,str(label_list[0]),str(column)]
            class_coord_tokens,bboxes,class_names = [],[],[]
            for i in range(1,len(label_list)):
                class_coords = label_list[i].split(',')
                coords = [float(class_coords[j]) for j in range(1,5)]
                bboxes.append(coords) # 这里的class_coord_tokens就是class_coords
                class_names.append(class_coords[0])
                class_coord_tokens.append([class_coords[0]] + [coords[j] for j in range(4)])

            class_coord_tokens,W_scaled,H_scaled = self.scale_coords(class_coord_tokens,W,H)
            class_coord_tokens = self.coordinates_encoding(class_coord_tokens)
            if np.random.random() > 0.6:
                class_coord_tokens,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled)
                prompt[0] = self.refine_token # 这个要改的
            
            train_select_len = np.random.choice(np.arange(1,len(label_list)),size=1)[0] # 至少是1，可能是满的
            train_select_idx = np.sort(np.random.choice(np.arange(1,len(label_list)),train_select_len,False))
            for i,idx in enumerate(train_select_idx):
                class_coords = class_coord_tokens[idx - 1]
                mask_len = np.random.choice(5,size=1)[0] # 至少有1个保证不会出现unconditional的状况
                mask_idx = np.random.choice(5,mask_len,False)
                cur_class_coord_prompt = []
                for j in range(5):
                    if j not in mask_idx:
                        cur_class_coord_prompt.append(str(class_coords[j]))
                prompt.append(' '.join(cur_class_coord_prompt))
        elif cond == 'R':
            prompt = [self.refine_token,layout_type,str(label_list[0]),str(column)]
            class_coord_tokens = []
            for i in range(1,len(label_list)):
                class_coords = label_list[i].split(',')
                coords = [float(class_coords[j]) for j in range(1,5)]
                class_coord_tokens.append([class_coords[0]] + [coords[j] for j in range(4)])
            class_coord_tokens,W_scaled,H_scaled = self.scale_coords(class_coord_tokens,W,H)
            class_coord_tokens = self.coordinates_encoding(class_coord_tokens)
            class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled)
            # class_coord_tokens_perturbed = self.coordinates_encoding(class_coord_tokens_perturbed)
            for i in range(len(class_coord_tokens_perturbed)):
                prompt.append(' '.join([str(item) for item in class_coord_tokens_perturbed[i]]))
        else:
            if np.random.random() > 0.5:
                prompt = [self.norefine_token,layout_type]
            else:
                prompt = self.generate_natural_prompt(layout_type,label_list[0],column)
            class_coord_tokens = []
            for i in range(1,len(label_list)):
                class_coords = label_list[i].split(',')
                coords = [float(class_coords[j]) for j in range(1,5)]
                class_coord_tokens.append([class_coords[0]] + [int(round(coords[j],0)) for j in range(4)])
            class_coord_tokens,W_scaled,H_scaled = self.scale_coords(class_coord_tokens,W,H) # 记得scale
            class_coord_tokens = self.coordinates_encoding(class_coord_tokens)

        prompt = self.bbox_sep_token.join(prompt)
        prompt = self.bos_token + prompt

        if cond == 'O': # add relation with a probability of 0.2
            if np.random.random() > 0.8:
                relation_tokens = self.generate_relation(bboxes,class_names)
                prompt = prompt + ',' + relation_tokens
        
        prompt = prompt + self.input_sep_token
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids']
        # print(prompt)
        # print(self.tokenizer.decode(prompt.squeeze()))
        sep_point = prompt.shape[1]
        prompt_after = []
        for i in range(len(class_coord_tokens)):
            prompt_after.append(' '.join([str(item) for item in class_coord_tokens[i]]))
        prompt_after = self.bbox_sep_token.join(prompt_after)
        prompt_after = self.tokenizer(prompt_after,return_tensors='pt')['input_ids']
        prompt = torch.cat((prompt,prompt_after),dim=1)
        label = torch.cat((prompt[0:1,1:],self.tokenizer(self.eos_token,return_tensors='pt')['input_ids']),dim=1)
        # print(self.tokenizer.decode(label.squeeze().numpy()))
        # label[0,:sep_point - 1] = -100
        prompt = prompt.squeeze()
        # print(self.tokenizer.decode(prompt.squeeze().numpy()))
        # print(self.tokenizer.decode(label.squeeze().numpy()))
        label = label.squeeze()
        return prompt,label,self.tokenizer.eos_token_id
    
    def encode_label(self,index):
        label_str = self.labels[index]
        print(index,self.names[index])
        label_list = label_str.split(';')
        W,H = [float(l) for l in label_list[-3].split(',')]
        column = int(label_list[-2])
        layout_type = label_list[-1]
        label_list = label_list[:-3]

        label = [self.norefine_token,layout_type,str(label_list[0]),str(column)]
        class_coord_tokens,bboxes,class_names = [],[],[]
        for i in range(1,len(label_list)):
            class_coords = label_list[i].split(',')
            coords = [float(class_coords[j]) for j in range(1,5)]
            bboxes.append(coords)
            class_names.append(class_coords[0])
            class_coord_tokens.append([class_coords[0]] + coords)
        class_coord_tokens,W_scaled,H_scaled = self.scale_coords(class_coord_tokens,W,H)
        class_coord_tokens = self.coordinates_encoding(class_coord_tokens)
        for i in range(len(class_coord_tokens)):
            coords = class_coord_tokens[i]
            cur_element_label = [coords[0]] + [str(coords[j]) for j in range(1,5)]
            label.append(' '.join(cur_element_label))
        return label,label_list,W_scaled,H_scaled,column,layout_type,bboxes,np.asanyarray(class_coord_tokens)
    
    def encode_test(self,index,cond):
        if 'U' in cond:
            return self.encode_test_unconditional(index,cond)
        elif 'R' in cond:
            return self.encode_test_refinement_mixed(index,cond)

        label,label_list,W_scaled,H_scaled,column,layout_type,bboxes,class_coord_tokens = self.encode_label(index) # 这个时候已经scale过坐标了
        class_names = class_coord_tokens[:,0].tolist() # bboxes保留一下吧，这是没有scale过坐标的，拿来算位置关系而已
        prompt = label[:4]
        test_select_len = np.random.choice(np.arange(1,len(label_list)),size=1)[0] # 至少是1，可能是满的
        test_select_idx = np.sort(np.random.choice(np.arange(0,len(class_coord_tokens)),test_select_len,False))
        if cond == 'C':
            for _,idx in enumerate(test_select_idx): # n个object，总共就是5*n个token
                class_coords = class_coord_tokens[idx]
                cur_class_coord_prompt = [str(e) for e in class_coords]
                prompt.append(' '.join(cur_class_coord_prompt))
        elif cond == 'T':
            for _,idx in enumerate(test_select_idx):
                prompt.append(class_coord_tokens[idx][0])
        elif cond == 'T-S':
            for _,idx in enumerate(test_select_idx):
                class_coords = class_coord_tokens[idx]
                cur_class_coord_prompt = [class_coords[0]] + [str(e) for e in class_coords[3:]]
                prompt.append(' '.join(cur_class_coord_prompt))
        elif cond == 'T-A': # 全部type都给
            for idx in range(len(class_coord_tokens)):
                prompt.append(class_coord_tokens[idx][0])
        elif cond == 'T-S-A': # 全部type都给
           for idx in range(len(class_coord_tokens)):
                cur_class_coord_prompt = [class_coord_tokens[idx][0]] + [str(e) for e in class_coord_tokens[idx][3:]]
                prompt.append(' '.join(cur_class_coord_prompt))
        elif cond == 'T-L-A': # 全部type都给，给relation
            for idx in range(len(class_coord_tokens)):
                prompt.append(class_coord_tokens[idx][0])
        elif cond == 'T-S-P': # 选出部分元素，T，xywh都随机给
            for _,idx in enumerate(test_select_idx): # n个object，总共就是5*n个token
                class_coords = class_coord_tokens[idx]
                mask_len = np.random.choice(5,size=1)[0] # 至少有1个保证不会出现unconditional的状况
                mask_idx = np.random.choice(5,mask_len,False)
                cur_class_coord_prompt = []
                for j in range(len(class_coords)):
                    if j not in mask_idx:
                        cur_class_coord_prompt.append(str(class_coords[j]))
                prompt.append(' '.join(cur_class_coord_prompt))

        # prompt.append(self.vocab[self.input_sep_token])
        prompt = self.bbox_sep_token.join(prompt)
        prompt = self.bos_token + prompt
        if cond == 'T-L-A':
            relation_tokens = self.generate_relation(bboxes,class_names) # 这里已经是字符串了
            prompt = prompt + ',' + relation_tokens
        prompt = prompt + self.input_sep_token
        # print(prompt)
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids'].squeeze()
        return prompt,label,layout_type,W_scaled,H_scaled,self.tokenizer(self.pad_token,return_tensors='pt')['input_ids'].squeeze()

    def encode_test_unconditional(self,index,cond): # cond是U或者是U-P
        label,label_list,W_scaled,H_scaled,column,layout_type,_,_ = self.encode_label(index)
        if cond == 'U':
            prompt = [self.norefine_token,layout_type]
        else:
            prompt = self.generate_natural_prompt(layout_type,label_list[0],column)
        prompt = self.bbox_sep_token.join(prompt)
        prompt = self.bos_token + prompt
        prompt = prompt + self.input_sep_token
        # print(prompt)
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids'].squeeze()
        return prompt,label,layout_type,W_scaled,H_scaled,self.tokenizer(self.pad_token,return_tensors='pt')['input_ids'].squeeze()

    def encode_test_refinement_mixed(self,index,cond): # 这里的refinement不是一定出现的，是按照概率和completion出现的，R必须得每个元素都出现
        label_str = self.labels[index]
        print(index,self.names[index])
        label_list = label_str.split(';') # label_list[0]不能是0
        W,H = [float(l) for l in label_list[-3].split(',')]
        column = int(label_list[-2])
        layout_type = label_list[-1]
        label_list = label_list[:-3]
        label = [self.refine_token,layout_type,str(label_list[0]),str(column)]
        
        class_coord_tokens = []
        for i in range(1,len(label_list)):
            class_coords = label_list[i].split(',')
            class_coord_tokens.append([class_coords[0]] + [float(class_coords[j]) for j in range(1,5)])
        class_coord_tokens,W_scaled,H_scaled = self.scale_coords(class_coord_tokens,W,H)
        class_coord_tokens = self.coordinates_encoding(class_coord_tokens)
        
        for i in range(len(class_coord_tokens)):
            coords = class_coord_tokens[i]
            cur_element_label = [coords[0]] + [str(coords[j]) for j in range(1,5)]
            label.append(' '.join(cur_element_label))

        prompt = label[:4]
        if cond == 'R': # 就这样setting先,本身就是C和PC
            class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled,p=1)
            for i in range(len(class_coord_tokens_perturbed)):
                prompt.append(' '.join([str(item) for item in class_coord_tokens_perturbed[i]]))
        else: # 训练用0.5是合理的，否则混合情况里面没法学习PCM的情况           
            test_select_len = np.random.choice(np.arange(1,len(label_list)),size=1)[0] # 至少是1，可能是满的
            test_select_idx = np.sort(np.random.choice(np.arange(1,len(label_list)),test_select_len,False))
            if cond == 'C-R-A': # 对应CM那种情况
                class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled,p=1)
                for _,idx in enumerate(test_select_idx): # n个object，总共就是5*n个token
                    prompt.append(' '.join([str(item) for item in class_coord_tokens_perturbed[idx - 1]]))
            elif cond == 'T-S-R': # T-S-R-A，对应CM那种情况
                class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled,p=1)
                for idx in test_select_idx:
                    class_coords = class_coord_tokens_perturbed[idx - 1]
                    cur_class_coord_prompt = [class_coords[0]]
                    for j in range(1,len(class_coords)):
                        if j in [3,4]:
                            cur_class_coord_prompt.append(str(class_coords[j]))
                    prompt.append(' '.join(cur_class_coord_prompt))
            elif cond == 'P-S-R':
                class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W,H,p=1)
                # for idx in range(1,len(label_list)):
                for idx in test_select_idx:
                    class_coords = class_coord_tokens_perturbed[idx - 1]
                    mask_len = np.random.choice(5,size=1)[0] # 至少有1个保证不会出现unconditional的状况
                    mask_idx = np.random.choice(np.arange(1,5),mask_len,False)
                    cur_class_coord_prompt = []
                    for j in range(1,len(class_coords)):
                        if j not in mask_idx:
                            cur_class_coord_prompt.append(str(class_coords[j]))
                    prompt.append(' '.join(cur_class_coord_prompt))
            elif cond == 'B-R': # 随便cxywh的组合,Box-Refinement，对应PCM那种情况
                class_coord_tokens_perturbed,refine_flags = self.add_perturbations_relative(class_coord_tokens,W_scaled,H_scaled,p=0.5)
                for _,idx in enumerate(test_select_idx): # n个object，总共就是5*n个token
                    class_coords = class_coord_tokens_perturbed[idx - 1]
                    mask_len = np.random.choice(5,size=1)[0] # 至少有1个保证不会出现unconditional的状况
                    mask_idx = np.random.choice(5,mask_len,False)
                    cur_class_coord_prompt = []
                    for j in range(len(class_coords)):
                        if j not in mask_idx:
                            cur_class_coord_prompt.append(str(class_coords[j]))
                    prompt.append(' '.join(cur_class_coord_prompt))

        prompt = self.bbox_sep_token.join(prompt)
        prompt = self.bos_token + prompt
        prompt = prompt + self.input_sep_token
        # print(prompt)
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids'].squeeze()
        return prompt,label,layout_type,W_scaled,H_scaled,self.tokenizer(self.pad_token,return_tensors='pt')['input_ids'].squeeze()

    def move_token(self,x,sep_point): # 这里不仅要把一开始的几个token弄掉，还要判断长度是不是正确
        # print(x,np.where(x[0,1:] == self.tokenizer.eos_token_id))
        stop_pred_point = np.sort(np.where(x[0,1:] == self.tokenizer.eos_token_id)[0])
        if len(stop_pred_point) == 0: # 还是没预测出来
            stop_pred_point = len(x[0]) + 1
        else:
            stop_pred_point = stop_pred_point[0] + 1
        start_pred_point = np.where(x[0] == self.input_sep_token_id)[0]
        if len(start_pred_point) > 0: # .#被一起编码了
            start_pred_point = start_pred_point[0] + 1
        else:
            start_pred_point = sep_point
        token_list = x[0,start_pred_point:stop_pred_point].tolist()
        token_natural = self.tokenizer.decode(token_list)
        token_natural = token_natural.split(';')
        return token_natural
    
    def decode_single(self,x,img_w,img_h,layout_type): # img_w和img_h是这张图片的宽高，用来做归一化，这个decode只对batch_size=1
        # x是list，已经分隔开元素了
        # print(x,len(x),img_w,img_h)
        whs = [int(img_w),int(img_h),int(img_w),int(img_h)]
        class_coords = []
        for i in range(len(x)):
            cur_pred = x[i].split(' ')
            if len(cur_pred) != 5 or not cur_pred[0] in self.classes[layout_type]:
                continue
            coord_strs = cur_pred[1:]
            coord_int,coord_flags = [],np.zeros((4,))
            for idx,e in enumerate(coord_strs):
                other_chars = re.compile('[^0-9]') # 匹配不是0~9的字符
                e = other_chars.sub('',e)
                if not e.isnumeric():
                    coord_flags[idx] = 1
                    continue
                e = int(e) - self.scale_max_len * idx
                if idx in [0,1]:
                    if not (0 <= e <= whs[idx]):
                        coord_flags[idx] = 1
                elif idx in [2,3]:
                    if e < 0:
                        coord_flags[idx] = 1
                    elif e > whs[idx]:
                        e = np.clip(e,0,whs[idx]) 
                coord_int.append(e)
            # print(x[i],coord_flags)
            if sum(coord_flags) > 0: # 证明有异常，大于长宽或者小于0
                continue
            class_coords.append(cur_pred[0:1] + coord_int)
        class_coords = np.array(class_coords)
        element_classes = np.array([self.classes2idx[layout_type][c] for c in class_coords[:,0]])
        coords = class_coords[:,1:]
        coords = np.array([list(map(lambda x:int(x),l)) for l in coords])
        bboxes = []
        for c in coords:
            x,y,w,h = c
            xc = x + w / 2
            yc = y + h / 2
            bboxes.append([xc / img_w,yc / img_h,w / img_w,h / img_h])
        bboxes = np.array(bboxes)
        bboxes_unnorm = coords
        return bboxes,bboxes_unnorm,element_classes

    def decode(self,x,img_w,img_h,layout_types): # 这里的x_mask和bbox_mask不是同一个东西，是给输出用的，确保解码出来的长度一致
        # if x.shape[0] == 1:
        bboxes,bboxes_unnorm,element_classes = self.decode_single(x,img_w[0],img_h[0],layout_types[0])
        mask = torch.ones((1,bboxes.shape[0]),dtype=torch.bool) # 这个batch要自己生成
        padding_mask = ~mask
        bboxes = bboxes[None,:] # bboxes的shape[0]必须是1，但是bboxes_unnorm不是1
        element_classes = element_classes[None,:]
        return bboxes,bboxes_unnorm,element_classes,mask,padding_mask
  
def collate_fn(batch):
    n = len(batch)
    prompt = [i[0] for i in batch]
    label = [i[1] for i in batch]
    pad_item = batch[0][-1]
    max_len = max([len(i) for i in prompt])
    prompt_padded = torch.full((n,max_len),fill_value=pad_item)
    label_padded = torch.full((n,max_len),fill_value=pad_item,dtype=torch.int64)
    for i,p in enumerate(prompt):
        prompt_padded[i,:len(p)] = p
        label_padded[i,:len(label[i])] = label[i]
    attention_mask = prompt_padded.ne(pad_item)
    return dict(input_ids=prompt_padded,labels=label_padded,attention_mask=attention_mask)

def test_collate_fn(batch):
    n = len(batch)
    prompt = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    layout_types = [i[2] for i in batch]
    W = [i[-3] for i in batch]
    H = [i[-2] for i in batch]
    pad_item = batch[0][-1]
    max_prompt_len = max([len(i) for i in prompt]) # label的max和prompt的max应该是一样的
    # max_label_len = max([len(i) for i in labels])
    prompt_padded = torch.full((n,max_prompt_len),fill_value=pad_item)
    # label_padded = torch.full((n,max_label_len),fill_value=pad_item,dtype=torch.int64)
    for i,p in enumerate(prompt):
        prompt_padded[i,:len(p)] = p
    attention_mask = prompt_padded.ne(pad_item)
    return prompt_padded,labels,layout_types,attention_mask,W,H