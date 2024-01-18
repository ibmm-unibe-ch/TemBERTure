from trainers import BERTTrainer

from transformers import  EarlyStoppingCallback
import wandb
from utils import logger


def Train(model_name, model_type, adapters, cls_train,cls_val,regr_train,regr_val,wandb_project, wandb_run_name, adapter_path=None):
    
    import random
    random.seed(10)
    print(random.random()) 
    
    from model import model_init
    tokenizer, model = model_init(model_type, 
                                  model_name, 
                                  adapters, 
                                  adapter_path)
    
    wandb.init(
        project= wandb_project,
        name = wandb_run_name)
    
    print(f'MODEL TYPE: {model_type}')

        
    if model_type == "BERT_cls" or model_type == "BERT_regr" or model_type == "BERTSequential" or model_type == "BERTInference":
        trainer = BERTTrainer(cls_train, 
                              cls_val, 
                              regr_train, 
                              regr_val, 
                              tokenizer, 
                              model, 
                              model_type, 
                              adapters)
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
    trainer.train(resume_from_checkpoint=True)
    results = trainer.evaluate()
    

    return logger.info(results)

