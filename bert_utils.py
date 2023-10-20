##########################################################################
########################### MODEL SET UP  ################################
##########################################################################

#https://github.com/Adapter-Hub/adapter-transformers/issues/248
from transformers import  BertAdapterModel, BertModel, PreTrainedModel, AutoConfig
import torch.nn as nn
import torch
import pandas as pd 

from data_prep import ClsData, RegrData, TrainerData
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
    #model.add_classification_head('SequentialAdapterBERT_adapter',layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    model.active_head == 'AdapterBERT_head_adapter' #pretrained for cls task adapter
    model.train_adapter(["AdapterBERT_adapter"])
    model = model.to('cuda')
    #model.set_active_adapters(["SequentialAdapterBERT_adapter"])
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



from main import DROPOUT

class BERT_regr(PreTrainedModel):
    
    def __init__(self,model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
         """

        self.bert =  BertModel.from_pretrained(model_name,config=config)

        # Instantiate the classifier head with some two-layer feed-forward classifier
        self.regressor =   nn.Sequential(
            nn.Dropout(p=DROPOUT, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.Tanh(),
            nn.Dropout(p=DROPOUT, inplace=False),
            nn.Linear(in_features=1024, out_features=1, bias=True),
        )


    def forward(self, input_ids, attention_mask,token_type_ids):
        ''' Feed input to BERT and the classifier to compute logits.
         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                       max_length)
         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                       information with shape (batch_size, max_length)
         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                       num_labels) '''
         # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
        
        #print('1',(((outputs[0].size())))) # for each aa values, [4, 512, 1024]
        #print('2',((outputs[0][:, 0, :].size()))) #cls token value [4, 1024]
         
        # Extract the last hidden state of the token `[CLS]` 
        last_hidden_state_cls = outputs[0][:, 0, :] #Here I’m taking the final hidden state of the [CLS] token, which serves as a good representation of an entire piece of text. 
        
 
         # Feed input to classifier to compute logits
        logits = self.regressor(last_hidden_state_cls)
 
        return logits

    

class BERT_cls(PreTrainedModel):
    def __init__(self,model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
         """

        self.bert =  BertModel.from_pretrained(model_name,config=config)

        # Instantiate the classifier head with some two-layer feed-forward classifier
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=1024, out_features=1, bias=True),
        )

    def forward(self, input_ids, attention_mask,token_type_ids):
        ''' Feed input to BERT and the classifier to compute logits.
         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                       max_length)
         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                       information with shape (batch_size, max_length)
         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                       num_labels) '''
         # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
        
        #print('1',(((outputs[0].size())))) # for each aa values, [4, 512, 1024]
        #print('2',((outputs[0][:, 0, :].size()))) #cls token value [4, 1024]
         
        # Extract the last hidden state of the token `[CLS]` 
        last_hidden_state_cls = outputs[0][:, 0, :] #Here I’m taking the final hidden state of the [CLS] token, which serves as a good representation of an entire piece of text. 
        
 
         # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
 
        return logits

    

