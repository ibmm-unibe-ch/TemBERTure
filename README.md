# TemBERTure_cls-regr-sequential-t5

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
