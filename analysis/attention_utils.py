#import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))

# libraries for the files in google drive
# General Libraries
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import gc

# Notebook Libraries
import math
from transformers import BertTokenizer, BertModel
import adapters
from adapters import BertAdapterModel
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel , PretrainedConfig

# ATTENTION SCORE FUNCTIONS

def TemBERTure_for_attention(best_model_path=None):
    print(f'MODEL WITH ADAPTER from {best_model_path}')
    model = BertAdapterModel.from_pretrained('Rostlab/prot_bert_bfd',output_attentions=True)
    if best_model_path is None:
        #!gdown --folder https://drive.google.com/drive/folders/1vg0Dyz8C2WVXh6fM6jnwnRXnmuRdP4-u #model weights folder
        #best_model_path='./content/final_adapter/'
        print('Model path not found')
    else:
        best_model_path = best_model_path
    #adapters.init(model)
    model.load_adapter(best_model_path+'AdapterBERT_adapter',with_head=False)
    model.load_head(best_model_path+'AdapterBERT_head_adapter')
    model.set_active_adapters(['AdapterBERT_adapter'])
    model.active_head == 'AdapterBERT_head_adapter'
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    #print(model)
    model_bert=model.bert
    return model_bert

def Rostlab_for_attention():
    print(f'MODEL from rostlab ')
    model = BertAdapterModel.from_pretrained('Rostlab/prot_bert_bfd',output_attentions=True)
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    #print(model)
    model_bert=model.bert
    return model_bert

## FROM SEQUENCE/FASTA FILE
def TemBERTure_for_classification(best_model_path=None):
    '''TemBERTure model with classification head.

    To be used in with hugginface pipelines:
    model_classification=TemBERTure_for_classification() # to compute classification score
    from transformers import pipeline
    pipe = pipeline("text-classification",model=model_classification,tokenizer='Rostlab/prot_bert_bfd')
    '''
    print(f'MODEL WITH ADAPTER from {best_model_path}')
    model = BertAdapterModel.from_pretrained('Rostlab/prot_bert_bfd',output_attentions=True)
    if best_model_path is None:
        #!gdown --folder https://drive.google.com/drive/folders/1qmH7VAxbYzWZdFFrv0VmgugAs49RKuWi #model weights folder
        #best_model_path='/content/TemBERTure_DEMO/final_adapter/'
        print('Model path not found')
    else:
        best_model_path = best_model_path
    #adapters.init(model)
    model.load_adapter(best_model_path+'AdapterBERT_adapter',with_head=False)
    model.load_head(best_model_path+'AdapterBERT_head_adapter')
    model.set_active_adapters(['AdapterBERT_adapter'])
    model.active_head == 'AdapterBERT_head_adapter'
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    model_classification=model
    return model_classification


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def attention_score_to_cls_token_and_to_all(input_text,model,device):

    ''' Retrieve attention from model outputs attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True
    is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer)
    of shape (batch_size, num_heads, sequence_length, sequence_length).

    Outputs:
    df_all_vs_all: pandas.Dataframe containing attention score for all tokens versus all tokens
    att_to_cls: pandas.Series with attention score of all tokens related to the CLS token
    df_att_to_cls_exp: att_to_cls in pandas.Dataframe format with score in exponential format

    '''
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
    #inputs = tokenizer(input_text, return_tensors='pt') ##COME ERA
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    attention_mask=inputs['attention_mask']
    outputs = model(inputs['input_ids'],attention_mask) #len 30 as the model layers #outpus.attentions
    attention = outputs[-1] #outpus has 2 dimensions, the second one are the attentions outputs.attentions
    inputs=inputs['input_ids'][0] #tokens id in the tokenizer vocab, same len as the input_test before adding whitespace and special char
    tokens = tokenizer.convert_ids_to_tokens(inputs)  # Convert input ids to token strings
    last_attentions=format_attention(attention, layers=[-1], heads=[-1]) #extract attention from last layer from last head (29,15)
    #(last_attentions[0][0]) #extracting from stacked list
    att_score=[]
    for i in list(last_attentions[0][0]): #extracting every list of attention
        #att_score.append((i).detach().numpy()) ### COME ERA
        att_score.append((i).detach().cpu().numpy())
        
    #len(att_score) # same len as number of tokens (len text + special char)
    x = np.stack(att_score, axis=0 )
    m = np.asmatrix(x) # attention score as matrix
    names = [_ for _ in tokens]
    df_all_vs_all = pd.DataFrame(m, index=names, columns=names) #attention score matrix all tokens vs all tokens
    #df_all_vs_all # df with attentio score all tokens vs all tokens
    att_to_cls = df_all_vs_all.loc['[CLS]'] #attention score only vs cls token
    #att_to_cls
    attention_to_cls_exp=[]

    att_to_cls.apply(lambda x: float(x))
    for i in att_to_cls:
        attention_to_cls_exp.append(math.exp(float(i)))

    token_att_num=tuple(zip(names,attention_to_cls_exp)) #pairs : token - a.score

    df_att_to_cls_exp=pd.DataFrame(token_att_num,columns=['token','attention'])
    #df_att_to_cls_exp['index'] = range(1, len(df_att_to_cls_exp) + 1) # starting count from 1 in the index
    df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[CLS]']
    df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[SEP]']
    #df_att_to_cls_exp

    torch.cuda.empty_cache()
    gc.collect()
    return df_all_vs_all,att_to_cls,df_att_to_cls_exp



def find_high_outliers_IQR(df):
    '''IQR is used to measure variability by dividing a data set into quartiles.
    The data is sorted in ascending order and split into 4 equal parts.
    The data points which fall below Q1 – 1.5 IQR or above Q3 + 1.5 IQR are outliers.
    IN this case we are only interested in 'high' outliers '''
    #df=pd.DataFrame(df['att_score'])
    q1=df['att_score'].quantile(0.25)
    q3=df['att_score'].quantile(0.75)
    IQR=q3-q1
    #outliers = df[((df['att_score']<(q1-1.5*IQR)) | (df['att_score']>(q3+1.5*IQR)))]
    outliers = df[((df['att_score']>(q3+1.5*IQR)))] #Keeping only 'high outlier'
    return outliers




def find_high_outliers_IQR_TrueFalse(df):
    '''IQR is used to measure variability by dividing a data set into quartiles.
    The data is sorted in ascending order and split into 4 equal parts.
    The data points which fall below Q1 – 1.5 IQR or above Q3 + 1.5 IQR are outliers.
    IN this case we are only interested in 'high' outliers '''
    #df=pd.DataFrame(df['att_score'])
    q1=df['att_score'].quantile(0.25)
    q3=df['att_score'].quantile(0.75)
    IQR=q3-q1
    #outliers = df[((df['att_score']<(q1-1.5*IQR)) | (df['att_score']>(q3+1.5*IQR)))]
    outliers = ((df['att_score']>(q3+1.5*IQR))) #Keeping only 'high outlier'
    return outliers

def attention_and_outliers(sequence,model,device):
    #computing attention respect to cls token
    _,att_to_cls,_ = attention_score_to_cls_token_and_to_all(' '.join(sequence),model,device)
    att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])
    # high attention extraction (True and False 1 per aa)
    df = pd.DataFrame({'aa':sequence, 'att_score': att_to_cls, })
    att_outliers=find_high_outliers_IQR_TrueFalse(df) #outlier = high attention 
    #updating the df
    return att_to_cls, att_outliers
