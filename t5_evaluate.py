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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from utils import formatter
import logging
test_logger = logging.getLogger("Logger1")
logger = logging.FileHandler('t5_test.log')
logger.setFormatter(formatter)
test_logger.addHandler(logger)
test_logger.setLevel(logging.INFO)


test_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "num_beams": 1,
    "max_length": 512,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
}



def test_evaluation(tasks, truth, preds):
    
    # Evaluating the tasks separately
    output_dict = {
        "binary classification": {"truth": [], "preds": [],},
        "regression": {"truth": [], "preds": [],},
    }

    #results_dict = {}

    for task, truth_value, pred in zip(tasks, truth, preds):
        output_dict[task]["truth"].append(truth_value)
        output_dict[task]["preds"].append(pred)
    
    for task, outputs in output_dict.items():
        if task == "binary classification":
            test_logger.info('')
            test_logger.info('---------------------------------------')
            test_logger.info('** BINARY CLASSIFICATION TASK **')
            test_logger.info('---------------------------------------')

            task_truth = [int(float(t)) for t in output_dict[task]["truth"]]
            task_preds = [int(float(p)) for p in output_dict[task]["preds"]]
            from evaluate import classification_test_performances_withpreds, regression_test_performances
            preds, predicted_labels, labels = classification_test_performances_withpreds(task_preds, task_truth)

        if task == "regression":
            test_logger.info('')
            test_logger.info('---------------------------------------')
            test_logger.info('** REGRESSION TASK **')
            test_logger.info('---------------------------------------')
            
            task_truth = [float(t) for t in output_dict[task]["truth"]]
            #print('task truth regression',task_truth)
            task_preds = [float(p) for p in output_dict[task]["preds"]]
            #print('task preds regression',task_preds)
            preds, true_value = regression_test_performances(task_truth, task_preds)
            
            

    #with open(f"preds_result_{datetime.now()}.json", "w") as f:
        #json.dump(output_dict, f)
        
    return True



def T5_evaluate_out(cls_test,regr_test,model_name_or_path, best_adapter_path,adapters):
    
    test_logger.info("-----------------------------------")
    test_logger.info("TEST Results: ")
    test_logger.info(f'Test data for classification task: {cls_test}')
    test_logger.info(f'Test data for regression task: {regr_test}')
    test_logger.info(f'Best model & tokenizer from path : {best_adapter_path}')
    test_logger.info('')
    
    # Load the trained model
    from t5_utils import T5, T5TestData
    #model, tokenizer = T5(adapters=adapters,model_name=model_name_or_path,eval=True,best_adapter_path=best_adapter_path)
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(best_adapter_path)
    model = T5ForConditionalGeneration.from_pretrained(best_adapter_path)
    if adapters:
        model.set_active_adapters(["T5_adapter"])
        test_logger.info('Adapters activation set to True')

    df, to_predict, truth, tasks = T5TestData(cls_test,regr_test)

    import tqdm

    preds = []
    
    for i in tqdm.tqdm(range(len(to_predict))):
        input_texts = to_predict[i]
        encoded = tokenizer(input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").input_ids.to("cuda")
        outputs = model.to("cuda").generate(encoded)
        preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
    preds = [float(x) for x in preds]

    # Taking only the first prediction
    #preds = [pred[0] for pred in preds]
    df["predicted"] = preds
    
    # Saving the predictions if needed
    df.to_csv(f"test_out.txt")
    
    return test_evaluation(tasks, truth, preds)












