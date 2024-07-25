from transformers import BertTokenizer
from adapters import BertAdapterModel
#import logging
import tqdm 
import math
import numpy as np
import torch.nn as nn
#logger = logging.getLogger(__name__)

class TemBERTure:
    """
    This class initializes and utilizes a pretrained BERT-based model (model_name) with adapter layers tuned
    for classification or regression tasks. The adapter path (adapter_path) provides the pre-trained
    adapter and head for the specified model and task (regression or classification).

    Attributes:
        adapter_path (str): Path to pre-trained adapters and heads for the model.
        model_name (str, default='Rostlab/prot_bert_bfd'): Name of the BERT-based model.
        batch_size (int, default=16): Batch size for predictions.
        device (str, default='cuda'): Device for running the model ('cuda' or 'cpu').

    Methods:
        __init__: Initializes the TemBERTure class with the specified BERT-based model,
                adapter path, tokenizer, batch size, and device.
        predict: Takes input texts, tokenizes them, and predicts outputs (classification/regression)
                using the loaded model and its adapters.
    """
    def __init__(self, adapter_path, model_name='Rostlab/prot_bert_bfd',batch_size=16, device='cuda', task = 'regression'):
        self.model = BertAdapterModel.from_pretrained(model_name) 
        self.model.load_adapter(adapter_path + 'AdapterBERT_adapter', with_head=True)
        self.model.load_head(adapter_path + 'AdapterBERT_head_adapter')
        self.model.set_active_adapters(['AdapterBERT_adapter'])
        self.model.active_head == 'AdapterBERT_head_adapter'  # pretrained for cls task adapter
        self.model.train_adapter(["AdapterBERT_adapter"])
        self.model.delete_head('default')
        self.model.bert.prompt_tuning = nn.Identity()
        #logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path}')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = device
        self.task = task
    
    def predict(self, input_texts):
        self.model = self.model.to(self.device)
        if not isinstance(input_texts, list):
            input_texts = [input_texts]
        input_texts = [" ".join("".join(sample.split())) for sample in input_texts]
        #input_texts = input_texts.tolist()
        print(input_texts)
        nb_batches = math.ceil(len(input_texts) / self.batch_size)
        y_preds = []

        for i in tqdm.tqdm(range(nb_batches)):
            batch_input = input_texts[i * self.batch_size: (i+1) * self.batch_size]
            encoded = self.tokenizer(batch_input, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            y_preds += self.model(**encoded).logits.reshape(-1).tolist()

        if self.task == 'classification':
            preds = 1 / (1 + np.exp(-np.array(y_preds)))
            y_preds = (preds > 0.5).astype(int)# Trasforma le probabilità in etichette binarie
            status = ['Thermophilic' if pred == 1 else 'Non-thermophilic' for pred in y_preds]
            print('Predicted thermal class:', status)
            print('Thermophilicity prediction score:', preds[0])
            
            return [status,preds]
        
        if self.task == 'regression':
            print('Predicted melting temperature:', y_preds)
            
            return y_preds

       
    
    
