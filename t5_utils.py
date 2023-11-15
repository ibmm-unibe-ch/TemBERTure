import pandas as pd 

from utils import logger 
from data_prep import binary_classification, concatenate_prefix_and_input, regression, TrainerData


def T5(adapters,model_name,eval=False,best_adapter_path=None):
    from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
    config_class, model_class =(T5Config,  T5ForConditionalGeneration)
    config = config_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if adapters == True:
        logger.info('YES ADAPTERS MODEL')
        model.add_adapter("T5_adapter")
        model.train_adapter(["T5_adapter"])
        model.set_active_adapters(["T5_adapter"])
    
    #if eval == True:
        #model.load_adapter(best_adapter_path+'T5_adapter')
        #model.set_active_adapters(["T5_adapter"])
        
    custom_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','.',':']
    custom_vocab.extend([str(i) for i in range(1, 11)])
    num_added_toks = tokenizer.add_tokens(custom_vocab)
    print("We have added", num_added_toks, "tokens to T5 model")
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


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


def T5TestData(cls_test,regr_test):
    print('** DATA MUST BE IN CSV FORMAT **')
    '''Note that a ": “ is inserted between the prefix and the input_text when preparing the data.
    This is done automatically when training but needs to be handled manually for prediction.'''
     
    ############################## binary classification data ##############################
    binary_test_df = pd.read_csv(cls_test, header=None,sep=',')
    
    binary_test_df = pd.DataFrame({
        'prefix': ["binary classification" for i in range(len(binary_test_df))],
        'input_text': binary_test_df[1].str.replace('\n', ' '),
        'target_text': binary_test_df[2].astype(int),
    })
 
    # prefix : input_text as required for multi task t5
    binary_test_df['input_text'] = binary_test_df.apply(concatenate_prefix_and_input, axis=1)
    #binary_test_df = binary_test_df.drop(columns=['prefix'])
    logger.info(binary_test_df[:2])
    print(binary_test_df[:2])
    
    ############################# regression data ##########################################
    
    regr_test_df = pd.read_csv(regr_test, header=None, sep=',')
    
    regr_test_df = pd.DataFrame({
        'prefix': ["regression" for i in range(len(regr_test_df))],
        'input_text': regr_test_df[1],
        'target_text': regr_test_df[3].astype(float), #regr data are in the format id,seq,cls_label,tm and in this case i want the tm as target
    })
    
    # prefix : input_text as required for multi task t5
    regr_test_df['input_text'] = regr_test_df.apply(concatenate_prefix_and_input, axis=1)
    #regr_train_df = regr_train_df.drop(columns=['prefix'])
    logger.info(regr_test_df[:2])
    print(regr_test_df[:2])

    ############################### TRAIN&EVAL DATA ########################################
    test_df = pd.concat([binary_test_df, regr_test_df]).reset_index(drop=True)
    
    test_df.to_csv("test_data.tsv", "\t")
    
    logger.info(f' ** MULTI-TASK TEST DATASET:{test_df}, # data {len(test_df)} **')
    
    from datasets import Dataset
    #test_df = Dataset.from_pandas(test_df)
    to_predict = test_df['input_text'].values.tolist()
    truth = test_df['target_text']
    tasks = test_df['prefix']
    
    return test_df,  to_predict, truth, tasks
