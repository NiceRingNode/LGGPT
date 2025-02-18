# -*- coding:utf-8 -*-

import logging
import pathlib,os
import transformers
from datasets import build_instruction_tuning_data_module
from transformers import GPT2LMHeadModel,GPT2Config,AutoModelForCausalLM,T5ForConditionalGeneration,AutoConfig,MT5ForConditionalGeneration
logger = logging.getLogger(__name__)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

class SupervisedInstructionTuningTrainer:
    """Trainer for supervised instruction tuning.

    Args:
        cfg (easydict): Training config.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.model.name == 'gpt2':
            gpt2xl_config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                bos_token_id=50256,
                eos_token_id=50256,
                pad_token_id=50256,
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
            model = GPT2LMHeadModel(config=gpt2xl_config)

        train_dataset,eval_dataset,data_collator = build_instruction_tuning_data_module(cfg)
        self.trainer = transformers.Trainer(
            model=model,
            tokenizer=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=cfg.output_dir,
                num_train_epochs=cfg.num_train_epochs,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                evaluation_strategy=cfg.evaluation_strategy,
                eval_steps=cfg.eval_steps,
                save_strategy=cfg.save_strategy,
                save_steps=cfg.save_steps,
                save_total_limit=cfg.save_total_limit,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_ratio=cfg.warmup_ratio,
                lr_scheduler_type=cfg.lr_scheduler_type,
                logging_strategy=cfg.logging_strategy,
                logging_steps=cfg.logging_steps,
                bf16=cfg.bf16,
                # tf32=cfg.tf32,
                fsdp=cfg.fsdp,
                fsdp_config=cfg.fsdp_config,
                deepspeed=cfg.deepspeed,
                gradient_checkpointing=cfg.gradient_checkpointing,
                optim="adamw_torch",
                report_to='none',
                # dataloader_pin_memory=False,
            ),
            data_collator=data_collator,
        )

    def train(self, resume=False):
        logger.info("Start training...")
        if resume:
            if list(pathlib.Path(self.cfg.output_dir).glob("checkpoint-*")):
                resume_model_path = transformers.trainer_utils.get_last_checkpoint(self.cfg.output_dir)
                logger.info(f'Resume from {resume_model_path}...')
                self.trainer.train(resume_from_checkpoint=True)
            else:
                logger.info(f'checkpoint-xxx not found in {self.cfg.output_dir}, training from scratch...')
                self.trainer.train()
        else:
            self.trainer.train()
        self.trainer.save_state()
        self.save_model_safely()
        logger.info('Training is done!')

    def save_model_safely(self):
        """Collects the state dict and dump to disk."""
        state_dict = self.trainer.model.state_dict()
        if self.trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            self.trainer._save(self.cfg.output_dir, state_dict=cpu_state_dict)  # noqa