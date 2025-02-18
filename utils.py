# -*- coding:utf-8 -*-

import logging,sys,time,os
from functools import lru_cache,reduce
from termcolor import colored
import torch
from safetensors.torch import load_file

def save_ckpt(epoch,model,optimizer,lr_scheduler,logger,save_folder,subname,save_path=None):
    if optimizer is None and lr_scheduler is None:
        save_state = {'model':model.state_dict(),'epoch':epoch}
    else:
        save_state = {'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict(),
                    'epoch':epoch}
    if save_path is None:
        save_path = f'{save_folder}/ckpt-{epoch}-{subname}.pth'
    logger.info(f'{save_path} saving checkpoint......')
    torch.save(save_state,save_path)
    logger.info(f'{save_path} successfully saved.')

def load_ckpt(model,pretrained_root,device,logger,optimizer=None,scheduler=None,mode='train',resume=False): 
    if mode == 'train':
        if resume:
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['lr_scheduler'])
            print(model.load_state_dict(state_dict['model']))
            start_epoch = state_dict['epoch'] + 1
            logger.info(f'mode: "{mode} + resume" {pretrained_root} successfully loaded.')
        else:
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
            print(model.load_state_dict(state_dict,strict=False))
            logger.info(f'mode: "{mode} + pretrained" {pretrained_root} successfully loaded.')
            start_epoch = 0
        return start_epoch
    else:
        if pretrained_root.endswith('.pth') or pretrained_root.endswith('.bin'):
            state_dict = torch.load(pretrained_root,map_location=device)
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            new_state_dict = {}
            for k in state_dict:
                new_k = k.replace('module.','')
                new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            # state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
            print(model.load_state_dict(state_dict,strict=True))
        else:
            sd = load_file(pretrained_root)
            print(model.load_state_dict(sd,strict=True))
        # model.load_state_dict(state_dict['model'])
        logger.info(f'mode: "{mode}" {pretrained_root} successfully loaded.')

@lru_cache()
def create_logger(log_root,name='',test=False):
    os.makedirs(f'{log_root}',exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    color_fmt = colored('[%(asctime)s %(name)s]','green') + colored('(%(filename)s %(lineno)d)','yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO) # 分布式的等级
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    date = time.strftime('%Y-%m-%d') if not test else time.strftime('%Y-%m-%d') + '-test'
    file_handler = logging.FileHandler(f'{log_root}/log-{date}.txt',mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger

def convert_xywh_to_ltrb(bbox):
    x1, y1, w, h = bbox
    # x1 = xc - w / 2
    # y1 = yc - h / 2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def convert_xcycwh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]