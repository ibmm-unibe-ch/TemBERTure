import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
import logging
import logging.handlers
import numpy as np
import sys
import logging
import datasets
from datasets import load_dataset
import os
import random
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification, Trainer, TrainingArguments, EvalPrediction
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import logging
import os
import pickle
from multiprocessing import Pool
from os import truncate
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from datasets import Dataset as HFDataset
import logging


logging.basicConfig(level=logging.INFO,
                   filename='log.log',
                   format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                   datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                              '%m-%d-%Y %H:%M:%S')


def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.

    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}


def compute_metrics_for_regression(eval_pred):
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    '''To compare with the classification model, let's also define a notion of "accuracy": For any score predicted by the regressor, let's round it (assign it to the closest integer) 
	and assume that is its predicted class. We compare the predicted class and the actual class to build the overall accuracy score.'''
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse,
            "mae": mae, 
            "r2": r2, 
            "accuracy": accuracy}



def compute_metrics_for_classification(eval_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
    logits, labels = eval_pred
    predicted_probs = 1 / (1 + np.exp(-logits))  # Applica la funzione sigmoide ai logits
    predicted_labels = (predicted_probs > 0.5).astype(int)  # Trasforma le probabilità in etichette binarie
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    mcc = matthews_corrcoef(labels, predicted_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def prep_compute_metrics_for_multitaskT5(tokenizer):
    from datasets import load_metric
    import numpy as np
    
    import evaluate

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):

        preds = [pred.strip() for pred in preds]

        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):

        preds, labels = eval_preds

        if isinstance(preds, tuple):

            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(decoded_preds)
        logger.info(decoded_preds)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(decoded_labels)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}
        


                
        return {"Val BLEU Score" : result,
                #"Val ROUGE Scores": rouge_scores,
                #"Val Perplexity" : perplexity.item()
                }
            
    
    return compute_metrics

    
    