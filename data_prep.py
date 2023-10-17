import pandas as pd
from utils import logger 


def binary_classification(cls_train,cls_val):
    ######################## BINARY CLASSIFICATION TASK DATA PREP #################################
    binary_train_df = pd.read_csv(cls_train, header=None,sep=',')
    binary_eval_df = pd.read_csv(cls_val, header=None,sep=',')

    logger.info('*** BINARY CLASSIFICATION TASK ***')
    logger.info(f'* train dataset: {cls_train}')
    logger.info(f'* # train data: {len(binary_train_df)}')
    logger.info(f'* val dataset: {cls_val}')
    logger.info(f'* # val data: {len(binary_eval_df)}')


    binary_train_df = pd.DataFrame({
        'prefix': ["binary classification" for i in range(len(binary_train_df))],
        'input_text': binary_train_df[1].str.replace('\n', ' '),
        'target_text': binary_train_df[2].astype(int),
    })


    binary_eval_df = pd.DataFrame({
        'prefix': ["binary classification" for i in range(len(binary_eval_df))],
        'input_text': binary_eval_df[1].str.replace('\n', ' '),
        'target_text': binary_eval_df[2].astype(int),
    })
    

    
    return binary_train_df,binary_eval_df


def regression(regr_train,regr_val):
    ######################## REGRESSION TASK DATA PREP #################################
    
    regr_train_df = pd.read_csv(regr_train, header=None, sep=',')
    regr_eval_df = pd.read_csv(regr_val,  header=None, sep=',')
    
    logger.info('*** REGRESSION TASK ***')
    logger.info(f'* train dataset: {regr_train}')
    logger.info(f'* # train data: {len(regr_train_df)}')
    logger.info(f'* val dataset: {regr_val}')
    logger.info(f'* # val data: {len(regr_eval_df)}')
    
    
    regr_train_df = pd.DataFrame({
        'prefix': ["regression" for i in range(len(regr_train_df))],
        'input_text': regr_train_df[1],
        'target_text': regr_train_df[3].astype(float), #regr data are in the format id,seq,cls_label,tm and in this case i want the tm as target
    })
    

    regr_eval_df = pd.DataFrame({
        'prefix': ["regression" for i in range(len(regr_eval_df))],
        'input_text': regr_eval_df[1],
        'target_text': regr_eval_df[3].astype(float),
    })
    
    
    return regr_train_df,regr_eval_df

def ClsData(cls_train,cls_val):
    print('** DATA MUST BE IN CSV FORMAT **')
    
    ############################## binary classification data ##############################
    binary_train_df,binary_eval_df = binary_classification(cls_train,cls_val)

    binary_train_df = binary_train_df.drop(columns=['prefix'])
    binary_eval_df = binary_eval_df.drop(columns=['prefix'])

    logger.info(' Task Dataset:')
    logger.info(binary_train_df[:2])
    logger.info(binary_eval_df[:2])
    print(binary_train_df[:2])
    print(binary_eval_df[:2])
    

    ############################### TRAIN&EVAL DATA ########################################
    train_df = binary_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = binary_eval_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df.to_csv("data/train.tsv", "\t")
    eval_df.to_csv("data/val.tsv", "\t")
    
    
    return train_df,eval_df

def RegrData(regr_train,regr_val):
    print('** DATA MUST BE IN CSV FORMAT **')
    
    ############################# regression data ##########################################
    regr_train_df,regr_eval_df = regression(regr_train,regr_val)
    
    regr_train_df = regr_train_df.drop(columns=['prefix'])
    regr_eval_df = regr_eval_df.drop(columns=['prefix'])
    
    logger.info(' Task Dataset:')
    logger.info(regr_train_df[:2])
    logger.info(regr_eval_df[:2])
    print(regr_train_df[:2])
    print(regr_eval_df[:2])
    
    ############################### TRAIN&EVAL DATA ########################################
    train_df = regr_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = regr_eval_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df.to_csv("data/train.tsv", "\t")
    eval_df.to_csv("data/val.tsv", "\t")
    
    
    return train_df,eval_df


# Tokenize the input text for both training and evaluation datasets
def tokenize_function(examples, tokenizer, max_length=512):
    inputs = tokenizer(
        examples["input_text"],
        truncation=True,  # Enable truncation if needed
        padding=True,     # Enable padding if needed
        max_length=max_length,
        return_tensors="pt",
    )
    inputs['labels'] = examples["target_text"]
    print(inputs['input_ids'])
    return inputs


# Tokenize the input text for both training and evaluation datasets
def tokenize_function_T5(examples, tokenizer, max_length=512):
    inputs = tokenizer(
        examples["input_text"],
        truncation=True,  # Enable truncation if needed
        padding='max_length',     # Enable padding if needed
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokenizer(examples["target_text"],
            truncation=True,  # Enable truncation if needed
            padding='max_length',     # Enable padding if needed
            max_length=10,
            return_tensors="pt",
        )
    inputs["labels"] = labels["input_ids"]
    #print('T5 labels',inputs["labels"])
    #print('T5 inputs',inputs["input_ids"])
    print(inputs)
    
    return inputs


# Define a custom function to concatenate prefix and input_text, for multi task t5
def concatenate_prefix_and_input(row):
    return row['prefix'] + ": " + row['input_text']



def TrainerDataT5(train_df, eval_df,tokenizer):
    from datasets import Dataset
    from data_prep import tokenize_function_T5
    
    train_df['target_text'] = train_df['target_text'].astype(str)
    eval_df['target_text'] = eval_df['target_text'].astype(str)
    
    train_df = Dataset.from_pandas(train_df).map(tokenize_function_T5, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=['input_text', 'target_text'])
    eval_df = Dataset.from_pandas(eval_df).map(tokenize_function_T5, fn_kwargs={"tokenizer": tokenizer},batched=True, remove_columns=['input_text', 'target_text'])
    
    columns_to_return = ['input_ids', 'labels', 'attention_mask']
    train_df.set_format(type='torch', columns=columns_to_return, device="cuda")
    eval_df.set_format(type='torch', columns=columns_to_return, device="cuda")
    
    logger.info(' ** DATASETS READY FOR TRAINING **')
    logger.info(train_df)
    logger.info(eval_df)
    
    return train_df, eval_df

def TrainerData(train_df, eval_df,tokenizer):
    from datasets import Dataset
    from data_prep import tokenize_function

    
    train_df = Dataset.from_pandas(train_df).map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=['input_text', 'target_text'])
    eval_df = Dataset.from_pandas(eval_df).map(tokenize_function, fn_kwargs={"tokenizer": tokenizer},batched=True, remove_columns=['input_text', 'target_text'])
    
    
    #columns_to_return = ['input_ids', 'labels', 'attention_mask']
    #train_df.set_format(type='torch', columns=columns_to_return, device="cuda")
    #eval_df.set_format(type='torch', columns=columns_to_return, device="cuda")
    
    logger.info(' ** DATASETS READY FOR TRAINING **')
    logger.info(train_df)
    logger.info(eval_df)
    
    return train_df, eval_df


        



    