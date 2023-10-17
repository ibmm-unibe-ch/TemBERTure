from trainers import T5Trainer, BERTTrainer

from transformers import  EarlyStoppingCallback,get_linear_schedule_with_warmup
import wandb
from utils import logger
import torch
#from main import SCHEDULER, WARMUP_RATIO


def Train(model_name, model_type, adapters, cls_train,cls_val,regr_train,regr_val,wandb_project, wandb_run_name,cls_adapter_path=None):
    
    from model import model_init
    tokenizer, model = model_init(model_type, model_name, adapters, cls_adapter_path)
    
    
    wandb.init(
        project= wandb_project,
        name = wandb_run_name)
    
    print(f'MODEL TYPE: {model_type}')
    
    if model_type == "T5":
        custom_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','.',':']
        custom_vocab.extend([str(i) for i in range(1, 11)])
        num_added_toks = tokenizer.add_tokens(custom_vocab)
        print("We have added", num_added_toks, "tokens to T5 model")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        trainer = T5Trainer(cls_train, cls_val, regr_train, regr_val, tokenizer, model)
        
    elif model_type == "BERT_cls" or model_type == "BERT_regr" or model_type == "BERTSequential":
        trainer = BERTTrainer(cls_train, cls_val, regr_train, regr_val, tokenizer, model, model_type, adapters)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    

    print('** MODEL PARAMS ** ')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable Parameters: ", trainable_params)
    print("Total Parameters: ", total_params)
    logger.info('** MODEL PARAMS ** ')
    logger.info(f"Trainable Parameters:  {trainable_params}")
    logger.info(f"Total Parameters: {total_params}")
    
    
    trainer.add_callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        

    trainer.train()
    results = trainer.evaluate()
    

    return logger.info(results)

