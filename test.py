# -*- coding:utf-8 -*-

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse,os,time,json
from datasets.dataset_unidata import Layout,test_collate_fn
from utils import create_logger,load_ckpt
from transformers import GPT2Config,GPT2LMHeadModel,AutoModelForCausalLM
from metrics import compute_alignment,compute_overlap,compute_maximum_iou,compute_docsim,FIDMultiModel

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--temp',type=float,default=0.)
parser.add_argument('--data',type=str,default='./data/publaynet/val.txt')
parser.add_argument('--ngpu',type=int,default=1)
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--rmp',type=str,default='',help='resume metrics path')
parser.add_argument('--dataset',type=str,default='publaynet')
parser.add_argument('--weights',type=str,default='./weights')
parser.add_argument('--output',type=str,default='./output')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--cond',type=str,default='C')
parser.add_argument('--name',type=str,default='')
parser.add_argument('--notes',type=str,default='')
opt = parser.parse_args()

opt.config_path = os.path.join(*opt.weights.split('/')[:3]) + '/'
opt.setting_path = os.path.join(*opt.weights.split('/')[:2]) + '/settings.json'
with open(opt.setting_path,'r',encoding='utf-8') as f:
    settings = json.loads(f.read())
opt.model = settings['model']
opt.model_config_path = settings['model_config_path']
opt.seed = settings['seed']
opt.name = settings['name']
opt.notes = settings['notes']
opt.log_root = settings['log_root']
# opt.gpu = settings['gpu']

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

os.makedirs(opt.output,exist_ok=True)

data_config = {
    'publaynet':{
        'data_path':'./data/publaynet/val_prompt.txt',
        'max_token_len':320
    },
    'rico':{
        'data_path':'./data/rico/test_prompt.txt',
        'max_token_len':500
    },
    'magazine':{
        'data_path':'./data/magazine/test_prompt.txt',
        'max_token_len':500
    }
}

fid_config = {
    'publaynet': {
        'max_token_len':320,
        'max_element_num':30,
        'fid_weight_path':'pretrained/fid_weights/FIDNetV3/publaynet-max25/model_best.pth.tar',
        'num_classes':5,
        'num_positions':25,
    },
    'rico': {
        'max_token_len':500,
        'max_element_num':50,
        'fid_weight_path':'pretrained/fid_weights/FIDNetV3/rico25-max25/model_best.pth.tar',
        'num_classes':25,
        'num_positions':25,
    },
    'magazine': {
        'max_token_len':320,
        'max_element_num':30,
        'fid_weight_path':'pretrained/fid_weights/FIDNetV3/magazine-max50/magazine.pth.tar',
        'num_classes':5,
        'num_positions':50,
    },
    'slide': {
        'max_token_len':320,
        'max_element_num':30,
        'fid_weight_path':'pretrained/fid_weights/FIDNetV3/slide-max25/model_best.pth',
        'num_classes':24,
        'num_positions':25,
    },
}

logger = create_logger(opt.log_root,name=opt.name,test=True)
config = data_config[opt.dataset]
opt.data = config['data_path']
test_dataset = Layout(config['data_path'],opt.dataset,tokenizer=opt.model,tokenizer_config=opt.model_config_path,train=False,condition=opt.cond)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=opt.batch_size,collate_fn=test_collate_fn)

gpt2_config = GPT2Config(
    vocab_size=test_dataset.vocab_size,
    n_positions=config['max_token_len'],
    n_embd=768,
    n_head=12,
    bos_token_id=1,
    eos_token_id=2,
)
gpt2xl_config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    bos_token_id=50256,
    eos_token_id=50256,
    attn_pdrop=0.1,
    n_ctx=1024,
    n_embd=1600,
    n_head=25,
    n_layer=48,
    resid_pdrop=0.1,
    summary_activation=None,
    summary_first_dropout=0.1,
    summary_proj_to_labels=True,
    summary_type='cls_index',
    summary_use_proj=True,
)
if opt.model == 'gpt2':
    model = GPT2LMHeadModel(config=gpt2xl_config)
    # model = GPT2LMHeadModel.from_pretrained(opt.config_path)
else:
    model = AutoModelForCausalLM.from_pretrained(opt.config_path)

if opt.cuda and torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    device = torch.device(f'cuda:{opt.gpu}')
else:
    device = torch.device('cpu')
model = model.to(device)

fid = FIDMultiModel(fid_config,device)

do_sample = True if opt.temp > 0. else False

logger.info(f'condition: {opt.cond}')
logger.info(config)

def resume_metrics(resume_metric_path,fid):
    records = torch.load(resume_metric_path,map_location=device)
    fid.load_features(records['fid']['fid_gt_feats'],records['fid']['fid_gen_feats'])
    gt_alignments = records['alignments']['gt_alignments']
    gen_alignments = records['alignments']['gen_alignments']
    gt_overlap = records['overlap']['gt_overlap']
    gen_overlap = records['overlap']['gen_overlap']
    gt_layouts = records['layouts']['gt_layouts']
    gen_layouts = records['layouts']['gen_layouts']
    sample_idx = records['sample_idx']
    return gt_alignments,gen_alignments,gt_overlap,gen_overlap,gt_layouts,gen_layouts,sample_idx

@torch.no_grad()
def test_impl(model):
    model = model.eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)

    gt_alignments = {}
    gen_alignments = {}
    gt_overlap = {}
    gen_overlap = {}
    gt_layouts = []
    gen_layouts = []

    if opt.rmp != '':
        gt_alignments,gen_alignments,gt_overlap,gen_overlap,gt_layouts,gen_layouts,record_idx = resume_metrics(opt.rmp,fid)
        logger.info('Loading recorded metrics successfully.')
    else:
        record_idx = -1

    try:
        total_time = 0
        total_s = time.time()
        for i,(inputs,labels,layout_types,attention_mask,W,H) in enumerate(test_loader):
            if i <= record_idx - 1: continue
            inputs = inputs.to(device)
            if layout_types[0] == 'App-UI':
                max_new_tokens = 450
            else:
                max_new_tokens = 300
            generate_config = {
                'input_ids':inputs,
                'max_new_tokens':max_new_tokens,
                'pad_token_id':test_dataset.pad_token_id,
                'do_sample':do_sample,
                # 'min_length': inputs.shape[1] + 120,
            }
            if do_sample:
                generate_config['temperature'] = opt.temp
            if opt.model in ['gpt2','qwen1.5','qwen2','tinyllamav1.1','nsfw','llama3']:
                generate_config['attention_mask'] = attention_mask.to(device)

            s = time.time()
            outputs = model.generate(**generate_config).cpu().numpy()
            outputs = test_dataset.move_token(outputs,len(inputs[0]))
            # print('moving:',outputs)
            gt_bboxes,_,gt_classes,gt_mask,gt_padding_mask = test_dataset.decode(labels[0][4:],img_w=W,img_h=H,layout_types=layout_types)
            gen_bboxes,_,gen_classes,gen_mask,gen_padding_mask = test_dataset.decode(outputs,img_w=W,img_h=H,layout_types=layout_types)
            gt_bboxes = torch.from_numpy(gt_bboxes).to(device).float()
            gt_classes = torch.from_numpy(gt_classes).to(device)
            gen_bboxes = torch.from_numpy(gen_bboxes).to(device).float()
            gen_classes = torch.from_numpy(gen_classes).to(device)

            gt_mask = gt_mask.to(device)
            gt_padding_mask = gt_padding_mask.to(device)
            gen_mask = gen_mask.to(device)
            gen_padding_mask = gen_padding_mask.to(device)

            fid.extract_features(gt_bboxes,gt_classes,gt_padding_mask,gt=True,data_type=layout_types[0])
            fid.extract_features(gen_bboxes,gen_classes,gen_padding_mask,gt=False,data_type=layout_types[0])     

            cur_gt_alignments = compute_alignment(gt_bboxes,gt_mask)
            for k,v in cur_gt_alignments.items():
                if gt_alignments.get(k) is None:
                    gt_alignments[k] = []
                gt_alignments[k].append(v)
            cur_gen_alignments = compute_alignment(gen_bboxes,gen_mask)
            for k,v in cur_gen_alignments.items():
                if gen_alignments.get(k) is None:
                    gen_alignments[k] = []
                gen_alignments[k].append(v)
            
            if layout_types[0] != 'App-UI' and layout_types[0] != 'slide':
                cur_gt_overlap = compute_overlap(gt_bboxes,gt_mask)
                for k,v in cur_gt_overlap.items():
                    if gt_overlap.get(k) is None:
                        gt_overlap[k] = []
                    gt_overlap[k].append(v)
                cur_gen_overlap = compute_overlap(gen_bboxes,gen_mask)
                for k,v in cur_gen_overlap.items():
                    if gen_overlap.get(k) is None:
                        gen_overlap[k] = []
                    gen_overlap[k].append(v)
        
            gt_layouts.append((gt_bboxes.cpu().numpy().squeeze(0),gt_classes.cpu().numpy().squeeze(0)))
            gen_layouts.append((gen_bboxes.cpu().numpy().squeeze(0),gen_classes.cpu().numpy().squeeze(0)))
            
            # inputs = test_dataset.tokenizer.decode(inputs.squeeze())
            # inputs = inputs[:-1].split(';')[4:]
            # input_bboxes,_,input_classes,mask,padding_mask = test_dataset.decode(inputs,layout_types=layout_types,img_w=W,img_h=H)
            # visualize_layout(input_bboxes[0],input_classes[0],layout_types[0],W[0],H[0],f'{opt.cond}-input{i}')
            # visualize_layout(gt_bboxes[0].cpu().numpy(),gt_classes[0].cpu().numpy(),layout_types[0],W[0],H[0],f'{opt.cond}-gt{i}')
            # visualize_layout(gen_bboxes[0].cpu().numpy(),gen_classes[0].cpu().numpy(),layout_types[0],W[0],H[0],f'{opt.cond}-gen{i}')
            # if i > 5:
            #     # break
            #     raise ValueError()
            # break

    except Exception as e:
        logger.info(f'Exception caught in mode {opt.cond}:',e)
        fid_gt_feats,fid_gen_feats = fid.features
        save_state = {}
        save_state['fid'] = {'fid_gt_feats':fid_gt_feats,'fid_gen_feats':fid_gen_feats}
        save_state['alignments'] = {'gt_alignments':gt_alignments,'gen_alignments':gen_alignments}
        save_state['overlap'] = {'gt_overlap':gt_overlap,'gen_overlap':gen_overlap}
        save_state['layouts'] = {'gt_layouts':gt_layouts,'gen_layouts':gen_layouts}
        save_state['sample_idx'] = i
        save_state['condition'] = opt.cond
        save_state['dataset'] = opt.dataset
        print(os.path.join(opt.config_path,f'{opt.cond}-sample{i}-metrics.pth'))
        torch.save(save_state,os.path.join(opt.config_path,f'{opt.cond}-sample{i}-metrics.pth'))

    finally:
        fid_gt_feats,fid_gen_feats = fid.features
        save_state = {}
        save_state['fid'] = {'fid_gt_feats':fid_gt_feats,'fid_gen_feats':fid_gen_feats}
        save_state['alignments'] = {'gt_alignments':gt_alignments,'gen_alignments':gen_alignments}
        save_state['overlap'] = {'gt_overlap':gt_overlap,'gen_overlap':gen_overlap}
        save_state['layouts'] = {'gt_layouts':gt_layouts,'gen_layouts':gen_layouts}
        save_state['sample_idx'] = i
        save_state['condition'] = opt.cond
        save_state['dataset'] = opt.dataset
        print(os.path.join(opt.config_path,f'{opt.cond}-sample{i}-metrics.pth'))
        torch.save(save_state,os.path.join(opt.config_path,f'{opt.cond}-sample{i}-metrics.pth'))

        fid_score = fid.compute_fid_score()
    
        for k,v in gt_alignments.items():
            gt_alignments[k] = (sum(v) / len(v)).cpu().item() * 100
        gt_alignments = '\t' + '\n\t'.join([f'{k}: {v:.4f}' for k,v in gt_alignments.items()])

        for k,v in gt_overlap.items():
            gt_overlap[k] = (sum(v) / len(v)).cpu().item() * 100
        gt_overlap = '\t' + '\n\t'.join([f'{k}: {v:.4f}' for k,v in gt_overlap.items()])

        for k,v in gen_alignments.items():
            gen_alignments[k] = (sum(v) / len(v)).cpu().item() * 100
        alignments = '\t' + '\n\t'.join([f'{k}: {v:.4f}' for k,v in gen_alignments.items()])

        for k,v in gen_overlap.items():
            gen_overlap[k] = (sum(v) / len(v)).cpu().item() * 100
        overlap = '\t' + '\n\t'.join([f'{k}: {v:.4f}' for k,v in gen_overlap.items()])

        s = time.time()
        max_iou = compute_maximum_iou(gt_layouts,gen_layouts)
        max_iou_time = time.time() - s

        s = time.time()
        doc_sim = compute_docsim(gt_layouts,gen_layouts)
        doc_sim_time = time.time() - s
        
        total_time = time.time() - total_s

        logger.info(f'test data root: {opt.data}\ntest loader length: {len(test_loader)} '
            f'test image number: {len(test_dataset)}\nmodel: {model.__class__.__name__}\n'
            f'condition: {opt.cond}\nbatch size: {opt.batch_size}\ndo sample: {do_sample}\n'
            f'temperature: {opt.temp}\nnotes: {opt.notes}')        
        logger.info(f'\nFID: {fid_score:.4f}\nAlignment:\n{alignments}\nOverlap:\n{overlap}\nMaximum IOU: {max_iou:.4f}\nDocSim: {doc_sim:.4f}')
        logger.info(f'GT Alignment:\n{gt_alignments}\nGT Overlap:\n{gt_overlap}')
        logger.info(f'Maximum IOU time: {max_iou_time:.2f}s')
        logger.info(f'DocSim time: {doc_sim_time:.2f}s')
        logger.info(f'total time: {total_time:.2f}s')
        logger.info(f'current sample index: {i}')

def test():
    load_ckpt(model,opt.weights,device,logger,mode='test')
    test_impl(model)

def main():
    test()

if __name__ == '__main__':
    main()