from utils import logger
from transformers import  AutoTokenizer
from bert_utils import *
from t5_utils import T5

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False




def model_init(model_type, model_name, adapters=None,adapter_path=None,):
    
    from main import DROPOUT
    
    if model_type == "T5":
        model = T5(adapters, model_name)
    elif model_type == "BERT_cls":
        if adapters:
            model = AdapterBERT_cls(model_name,dropout_p=DROPOUT)
        else:
            model = BERT_cls(model_name,dropout_p=DROPOUT)
    elif model_type == "BERT_regr":
        if adapters:
            model = AdapterBERT_regr(model_name,dropout_p=DROPOUT)
        else:
            model = BERT_regr(model_name)
    elif model_type == "BERTSequential" and adapters:
        model = TrainedAdapterBERT(model_name,adapter_path)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
     
    logger.info(f' * USING MODEL NAME/PATH: {model_name}')
    logger.info(f' * USING MODEL TYPE: {model_type}')
    
    
    logger.info(model)
        
    return tokenizer, model
            
            
            
    




