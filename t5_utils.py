import pandas as pd 

from utils import logger 
from data_prep import binary_classification, concatenate_prefix_and_input, regression, TrainerData


def T5(adapters,model_name):
    from transformers import T5Config, T5ForConditionalGeneration
    config_class, model_class =(T5Config,  T5ForConditionalGeneration)
    config = config_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, config=config)
    logger.info('NO ADAPTERS MODEL')

    if adapters == True:
        logger.info('YES ADAPTERS MODEL')
        model.add_adapter("T5_adapter")
        model.train_adapter(["T5_adapter"])
        model.set_active_adapters(["T5_adapter"])

    return model


def T5Data(cls_train,cls_val,regr_train,regr_val):
    print('** DATA MUST BE IN CSV FORMAT **')
    
    ############################## binary classification data ##############################
    binary_train_df,binary_eval_df = binary_classification(cls_train,cls_val)
    
    # prefix : input_text as required for multi task t5
    binary_train_df['input_text'] = binary_train_df.apply(concatenate_prefix_and_input, axis=1)
    binary_train_df = binary_train_df.drop(columns=['prefix'])
    binary_eval_df['input_text'] = binary_eval_df.apply(concatenate_prefix_and_input, axis=1)
    binary_eval_df = binary_eval_df.drop(columns=['prefix'])

    logger.info(binary_train_df[:2])
    logger.info(binary_eval_df[:2])
    print(binary_train_df[:2])
    print(binary_eval_df[:2])
    
    ############################# regression data ##########################################
    regr_train_df,regr_eval_df = regression(regr_train,regr_val)
    
    # prefix : input_text as required for multi task t5
    regr_train_df['input_text'] = regr_train_df.apply(concatenate_prefix_and_input, axis=1)
    regr_train_df = regr_train_df.drop(columns=['prefix'])
    regr_eval_df['input_text'] = regr_eval_df.apply(concatenate_prefix_and_input, axis=1)
    regr_eval_df = regr_eval_df.drop(columns=['prefix'])
    
    logger.info(regr_train_df[:2])
    logger.info(regr_eval_df[:2])
    print(regr_train_df[:2])
    print(regr_eval_df[:2])
    
    ############################### TRAIN&EVAL DATA ########################################
    train_df = pd.concat([binary_train_df, regr_train_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = pd.concat([binary_eval_df, regr_eval_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df.to_csv("data/train.tsv", "\t")
    eval_df.to_csv("data/val.tsv", "\t")
    
    logger.info(f' ** MULTI-TASK TRAIN DATASET:{train_df}, # data {len(train_df)} **')
    logger.info(f' ** MULTI-TASK VAL DATASET:{eval_df}, # data {len(eval_df)} **')
    
    
    return train_df,eval_df


