# -*- coding:utf-8 -*-

from datasets.dataset_unidata import Layout,collate_fn

def build_instruction_tuning_data_module(cfg):
    train_dataset = Layout(cfg.train_datasets.path,tokenizer=cfg.model.name,
                           tokenizer_config=cfg.model.model_path,train=True)
    eval_dataset = None
    data_collator = collate_fn
    return train_dataset,eval_dataset,data_collator
