
This repository provides the code used to implementate both tasks. To use TemBERTure please refere to the main [README](https://github.com/ibmm-unibe-ch/TemBERTure/blob/main/README.md) .

# 1. TemBERTure_CLS 

TemBERTureCLS is a sequence-based classifier that utilizes ProtBert and adopts an adapter-based approach, following the methodology proposed by Pfeiffer et al. (2020) and Houlsby et al. (2019). This classifier is fine-tuned to predict specific protein thermal category based on sequence data. The model has been trained three times to ensure robustness, but only the weights of the model leveraged for the results are reported here.

# 2. TemBERTure_Tm 
TemBERTure_Tm serves as a regression model designed to inference protein melting temperatures (Tm) derived solely from protein sequences.
This repository provides implementations and weights for both tasks, allowing users to leverage these models for various protein-related predictive tasks.
We tested different methodologies,also leveraging the classifier weights,and the code is reported below. For each model configurations we trained 3 replicas. However here are reported only the weights of best performing models replicas, that is the standard approach (randomly initialization).

TemBERTure_Tm serves as a regression model designed to infer protein melting temperatures (Tm) derived solely from protein sequences. We tested different methodologies, also leveraging the classifier weights, and the code is reported below. For each model configuration, we trained three replicas. However, only the weights of the best-performing model replicas are reported here, that are using the standard approach (random initialization).


#### Methodologies:

- **Sequential Approach:** 5 * 3 model replicas = 15 models
  - Exploits the pre-trained weights of the classifier in different stages of training.
  - Fine-tunes the model's pre-trained weights specifically for regression tasks.

- **Standard Approach:** 3 model replicas = 3 models
  - Directly fine-tunes the model on Tm data, bypassing classification training.
 
![Idea owner(25)](https://github.com/Ch-rode/TemBERTure/assets/61243245/07a15e0a-bc73-4164-9f13-1650eabbcb0e)


# 3. TRAINING:
### Arguments
```
import argparse
parser = argparse.ArgumentParser(description='')

# mode
parser.add_argument("--do_train", help="", type=bool,default=False)
parser.add_argument("--do_test", help="", type=bool,default=False)
parser.add_argument("--do_inference", help="", type=bool,default=False)

# model
parser.add_argument("--model_name_or_path", help="Pre-trained model to use for training or best model path for inference/test", type=str,required=False) #"ElnaggarLab/ankh-base"
parser.add_argument("--model_type", help="T5, Bert etc", type=str,required=False) 

# training data
parser.add_argument("--cls_train", help="", type=str,required=False)
parser.add_argument("--cls_val", help="", type=str,required=False)
parser.add_argument("--regr_train", help="", type=str,required=False)
parser.add_argument("--regr_val", help="", type=str,required=False)

# wandb training 
parser.add_argument("--wandb_project", help="", type=str,required=False,default='./test')
parser.add_argument("--wandb_run_name", help="", type=str,required=False,default=None)

# training args
parser.add_argument("--with_adapters", help="", type=bool,default=None)
parser.add_argument("--lr", help="", type=float,required=False,default=1e-5)
parser.add_argument("--weight_decay", help="", type=float,required=False,default=0.0)
parser.add_argument("--warmup_ratio", help="", type=float,required=False, default= 0)
parser.add_argument("--head_dropout", help="", type=float,required=False, default= 0.1)
parser.add_argument("--per_device_train_batch_size", help="", type=int,required=False, default= 16)
parser.add_argument("--per_device_eval_batch_size", help="", type=int,required=False, default= 16)

#parser.add_argument("--resume_from_checkpoint", help="", type=str,required=False, default = None) #Non funziona con TRainer e gli adapters, ho caricato gli ultimi checkpoint dell'ultima epoca e faccio un training da li usando BertSequential


# if SequentialBERT & test
parser.add_argument("--best_model_path", help="", type=str,required=False,default=None)

# if test
parser.add_argument("--test_data", help="", type=str,required=False,default=None)
parser.add_argument("--task", help="", type=str,required=False,default=None)


#if inference
parser.add_argument("--data", help="", type=str,required=False,default=None)
parser.add_argument("--column_to_predict", help="", type=int,required=False,default=None)
```
### Train the model 
```
python ./main.py \
--do_train True \
--model_name_or_path "Rostlab/prot_bert_bfd" \
--with_adapters True \
--cls_train "/ClassifierData/classifier_train_filtered" \
--cls_val "./ClassifierData/classifier_val_filtered" \
--regr_train "./RegressionData/regression_train_UpDownSampling" \
--regr_val "./RegressionData/regression_val" \
--wandb_project 'TemBERTure' \
--wandb_run_name 'run_name' \
--lr 1e-3 \
--head_dropout 0.2 \
```

* BERTClassifier:
```
--model_type 'BERT_cls' 
```

* BERTRegressor:
```
--model_type 'BERT_regr' 
```

* BERTSequential:
```
--best_model_path./BERT_cls/adapters/lr1e-5_headdrop02_linearwithwarmup/output/checkpoint-30755/
--model_type 'BERTSequential' 
```

# 4. TESTING 
```
python ./model/code/main.py \
--do_test True \
--model_name_or_path "Rostlab/prot_bert_bfd" \
--with_adapters True \
--test_data "./ClassifierData/classifier_test_filtered" \
--best_model_path ./BERT_cls/adapters/BEST_MODEL/lr_1e-5_headropout01/output/best_model_epoch4/
```
* Task to select the correct test dataset format:  
`--task classification`  
`--task regression`  
`--task regression_on_classification_data`  
`--task bacdive_sequence_classification`  
`--task classification_on_regression_data`  


