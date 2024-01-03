import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
import logging
import logging.handlers
import numpy as np
import sys
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification, Trainer, TrainingArguments, EvalPrediction
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from multiprocessing import Pool
from os import truncate
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer


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
    
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    
    
    return {"mse": mse,
            "mae": mae, 
            "r2": r2, 
            }



def compute_metrics_for_classification(eval_pred):
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
    
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


    