# -*- coding:utf-8 -*-

import os,argparse,json,time
from trainer import SupervisedInstructionTuningTrainer
from omegaconf import OmegaConf

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

data_config = {
    'unified':{
        'data_path':'./data/unified/train.txt',
        'max_token_len':500,
        'max_element_num':50,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--epochs',type=int,default=5)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--weight_decay',type=float,default=1e-2)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--device', default='cuda') # 这参数不用管，系统会自动分配
parser.add_argument('--dataset',type=str,default='magazine')
parser.add_argument('--save_freq',type=int,default=200)
parser.add_argument('--log_freq',type=int,default=300)
parser.add_argument('--pretrained_root',type=str,default='')
parser.add_argument('--weights_root',type=str,default='./weights')
parser.add_argument('--output_root',type=str,default='./output')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--config',type=str,default='./config/gpt2.yaml')
parser.add_argument('--name',type=str,default='LGGPT')
parser.add_argument('--notes',type=str,default='')
opt = parser.parse_args()
config = OmegaConf.load(opt.config)

date_str = f"{opt.name}-{time.strftime('%Y%m%d-%H%M%S')}"
opt.weights_root = f'{opt.weights_root}/{date_str}' # 这个地方只会初始化一次
config.output_dir = opt.weights_root
opt.log_root = f"{opt.log_root}/{time.strftime('%Y-%m-%d')}/{date_str}"
config.log_root = opt.log_root
opt.model = config.model.name
opt.model_config_path = config.model.model_path # Tokenizer也是用的这个路径
config.train_datasets.name = opt.dataset
config.train_datasets.path = data_config[opt.dataset]['data_path']
config.train_datasets.max_element_num = data_config[opt.dataset]['max_element_num']
config.num_train_epochs = opt.epochs
config.seed = opt.seed

os.makedirs(opt.weights_root,exist_ok=True)
os.makedirs(opt.log_root,exist_ok=True)

with open(f'{opt.weights_root}/settings.json','w',encoding='utf-8') as f:
    f.write(json.dumps({**vars(opt)},indent=4,ensure_ascii=False))

trainer = SupervisedInstructionTuningTrainer(config)
trainer.train()
