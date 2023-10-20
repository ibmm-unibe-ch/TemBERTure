# TemBERTure_cls-regr-sequential-t5
```
import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument("--do_train", help="", type=bool,default=False)
parser.add_argument("--do_test", help="", type=bool,default=False)
parser.add_argument("--do_inference", help="", type=bool,default=False)

parser.add_argument("--model_name_or_path", help="Pre-trained model to use for training or best model path for inference/test", type=str,required=False) #"ElnaggarLab/ankh-base"
parser.add_argument("--model_type", help="T5, Bert etc", type=str,required=False) 
parser.add_argument("--with_adapters", help="", type=bool,default=None)

parser.add_argument("--cls_train", help="", type=str,required=False)
parser.add_argument("--cls_val", help="", type=str,required=False)
parser.add_argument("--regr_train", help="", type=str,required=False)
parser.add_argument("--regr_val", help="", type=str,required=False)


parser.add_argument("--wandb_project", help="", type=str,required=False,default='./test')
parser.add_argument("--wandb_run_name", help="", type=str,required=False,default=None)

parser.add_argument("--lr", help="", type=float,required=False,default=1e-5)
parser.add_argument("--weight_decay", help="", type=float,required=False,default=0.0)
#parser.add_argument("--scheduler", help="", type=str,required=False,default = 'linear')
parser.add_argument("--warmup_ratio", help="", type=float,required=False, default= 0)
parser.add_argument("--head_dropout", help="", type=float,required=False, default= 0.1)
parser.add_argument("--per_device_train_batch_size", help="", type=int,required=False, default= 16)
parser.add_argument("--per_device_eval_batch_size", help="", type=int,required=False, default= 16)

# if SequentialBERT & test
parser.add_argument("--best_model_path", help="", type=str,required=False,default=None)

# if test
parser.add_argument("--test_data", help="", type=str,required=False,default=None)
parser.add_argument("--task", help="", type=str,required=False,default=None)
parser.add_argument("--test_out_path", help="", type=str,required=False,default='./test')
```

```
python ./main.py \
--do_train True \
--model_name_or_path "Rostlab/prot_bert_bfd" \
--with_adapters True \
--cls_train "/ibmm_data/TemBERTure/MultiTaskDataset/FinalDataset/ClassifierData/classifier_train_filtered" \
--cls_val "/ibmm_data/TemBERTure/MultiTaskDataset/FinalDataset/ClassifierData/classifier_val_filtered" \
--regr_train "/ibmm_data/TemBERTure/MultiTaskDataset/FinalDataset/RegressionData/regression_train_UpDownSampling" \
--regr_val "/ibmm_data/TemBERTure/MultiTaskDataset/FinalDataset/RegressionData/regression_val" \
--wandb_project 'TemBERTure' \
--wandb_run_name 'test1' \
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
--cls_adapter_path /ibmm_data/TemBERTure/model/BERT_cls/adapters/lr1e-5_headdrop02_linearwithwarmup/output/checkpoint-30755/
--model_type 'BERTSequential' 
```
