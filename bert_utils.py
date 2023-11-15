##########################################################################
########################### MODEL SET UP  ################################
##########################################################################

#https://github.com/Adapter-Hub/adapter-transformers/issues/248
from transformers import  BertAdapterModel, BertModel
import torch.nn as nn
from utils import logger


######## MODEL
def AdapterBERT_regr(model_name,dropout_p=0.4,n_head_layers=2,head_act_function='relu'):
    logger.info('YES ADAPTERS MODEL')
    model = BertAdapterModel.from_pretrained(model_name)           #'Rostlab/prot_bert_bfd') 
    model.add_adapter("AdapterBERT_adapter",set_active=True)
    model.add_classification_head('AdapterBERT_head_adapter',layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    model.active_head == 'AdapterBERT_head_adapter'
    model.train_adapter(["AdapterBERT_adapter"])
    model.set_active_adapters(["AdapterBERT_adapter"])
    logger.info(f' * USING HEAD DROPOUT: {dropout_p}')
    return model 


def AdapterBERT_cls(model_name,dropout_p=0.1,n_head_layers=2,head_act_function='tanh'):
    model = BertAdapterModel.from_pretrained(model_name) 
    logger.info('YES ADAPTERS MODEL')
    model.add_adapter("AdapterBERT_adapter",set_active=True)
    model.add_classification_head('AdapterBERT_head_adapter',num_labels=1, layers=n_head_layers,activation_function=head_act_function, dropout = dropout_p)
    model.active_head == 'AdapterBERT_head_adapter'
    model.train_adapter(["AdapterBERT_adapter"])
    model.set_active_adapters(["AdapterBERT_adapter"])
    logger.info(f' * USING HEAD DROPOUT: {dropout_p}')
    return model 
    
    
def TrainedAdapterBERT(model_name,adapter_path):
    model = BertAdapterModel.from_pretrained(model_name) 
    model.load_adapter(adapter_path+'AdapterBERT_adapter',with_head=True)
    model.load_head(adapter_path + 'AdapterBERT_head_adapter')
    model.active_head == 'AdapterBERT_head_adapter' #pretrained for cls task adapter
    model.train_adapter(["AdapterBERT_adapter"])
    logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path}')
    model = model.to('cuda')
    return model 


def AdapterBERT_sequential(model_name,adapter_path,dropout_p=0.4,n_head_layers=2,head_act_function='relu'):
    model = BertAdapterModel.from_pretrained(model_name) 
    model.load_adapter(adapter_path+'AdapterBERT_adapter',with_head=True)
    # using pretrained head weights:
    # model.load_head(adapter_path + 'AdapterBERT_head_adapter')
    model.add_classification_head('AdapterBERT_head_adapter',layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    model.active_head == 'AdapterBERT_head_adapter' #pretrained for cls task adapter
    model.train_adapter(["AdapterBERT_adapter"])
    logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path}')
    return model 
    


def BERT_cls(model_name,n_labels=None):
    logger.info('NO ADAPTERS MODEL')
    model = BertModel.from_pretrained(model_name) 
    classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=False),
        nn.Linear(in_features=1024, out_features=1024, bias=True),
        nn.Tanh(),
        nn.Dropout(p=0.1, inplace=False),
        nn.Linear(in_features=1024, out_features=n_labels, bias=True),
    )

    model.classifier = classifier
    
    return model 

