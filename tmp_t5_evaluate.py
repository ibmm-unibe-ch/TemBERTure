import json
from datetime import datetime
from pprint import pprint
from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
#from simpletransformers.t5 import T5Model
from sklearn.metrics import accuracy_score, f1_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
import os

from model import T5ModelForMT


'''Note that a ": “ is inserted between the prefix and the input_text when preparing the data.
 This is done automatically when training but needs to be handled manually for prediction.'''

def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


test_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "num_beams": 1,
    "max_length": 512,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
}


def test_data(data):
    # Load the test data
    df = pd.read_csv(data,sep=',').astype(str)
    print('** TEST DATA MUST BE IN CSV FORMAT WITH HEADER: PREFIX, SEQ, TARGET ***')
    # Prepare the data for testing
    to_predict = [
        prefix + ": " + input_text
        for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
    ]
    truth = df["target_text"].tolist()
    tasks = df["prefix"].tolist()
    
    return df, to_predict, truth, tasks    


def test_evaluation(tasks, truth, preds,test_out_path):
    
    # Evaluating the tasks separately
    output_dict = {
        "binary classification": {"truth": [], "preds": [],},
        "multilabel classification": {"truth": [], "preds": [],},
        "similarity": {"truth": [], "preds": [],},
    }

    results_dict = {}

    for task, truth_value, pred in zip(tasks, truth, preds):
        output_dict[task]["truth"].append(truth_value)
        output_dict[task]["preds"].append(pred)

    print(output_dict['binary classification']["preds"])
    print("-----------------------------------")
    print("Results: ")
    for task, outputs in output_dict.items():
        if task == "binary classification":

            task_truth = [int(float(t)) for t in output_dict[task]["truth"]]
            print('task truth cls',task_truth)
            task_preds = [int(float(p)) for p in output_dict[task]["preds"]]
            print('task preds cls',task_preds)
            results_dict[task] = {
                "F1 Score": f1_score(task_truth, task_preds),
                "Accuracy Score": accuracy_score(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {results_dict[task]['F1 Score']}")
            print(f"Accuracy Score: {results_dict[task]['Accuracy Score']}")
            print()

        if task == "regression":
            task_truth = [float(t) for t in output_dict[task]["truth"]]
            print('task truth regression',task_truth)
            print(output_dict[task]["preds"])
            task_preds = [float(p) for p in output_dict[task]["preds"]]
            print('task preds regression',task_preds)
            results_dict[task] = {
                "Pearson Correlation": pearson_corr(task_truth, task_preds),
                "Spearman Correlation": spearman_corr(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"Pearson Correlation: {results_dict[task]['Pearson Correlation']}")
            print(f"Spearman Correlation: {results_dict[task]['Spearman Correlation']}")
            print()

    with open(f"{test_out_path}/result_{datetime.now()}.json", "w") as f:
        json.dump(results_dict, f)
        
    return True




def do_test(test_out_path,data,model_name_or_path, best_adapter_path,adapters):
    # Load the trained model

    model = T5ModelForMT("t5", model_name_or_path, args=test_args, eval=True, best_adapter_path = best_adapter_path, adapters=adapters)
    df, to_predict, truth, tasks = test_data(data)
    
    # Get the model predictions
    preds = model.predict(to_predict)

    # Taking only the first prediction
    #preds = [pred[0] for pred in preds]
    df["predicted"] = preds
    
    if not os.path.exists(test_out_path):
        # Create a new directory because it does not exist
        os.makedirs(test_out_path)

    # Saving the predictions if needed
    df.to_csv(f"{test_out_path}/predictions_{datetime.now()}.txt")
    
    return test_evaluation(tasks, truth, preds,test_out_path)












