##########################################################################
########################### MODEL SET UP  ################################
##########################################################################

#https://github.com/Adapter-Hub/adapter-transformers/issues/248
#from adapters import  BertAdapterModel
#from transformers import BertAdapterModel
from adapters import BertAdapterModel
import torch.nn as nn
from utils import logger


######## MODEL
def AdapterBERT_regr(model_name,dropout_p=0.4,n_head_layers=2,head_act_function='relu'):
    logger.info('YES ADAPTERS MODEL')
    model = BertAdapterModel.from_pretrained(model_name)           #'Rostlab/prot_bert_bfd') 
    ## with adapter-hub
    model.add_adapter("AdapterBERT_adapter",set_active=True)
    #model.add_classification_head('AdapterBERT_head_adapter',layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    ## tmp try with custom head
    #model.register_custom_head("AdapterBERT_head_adapter", CustomHead)
    #model.add_custom_head(head_type="AdapterBERT_head_adapter", head_name="AdapterBERT_head_adapter",layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    #model.add_custom_head(head_type="AdapterBERT_head_adapter", head_name="AdapterBERT_head_adapter")
    ## with adapters library
    config = dict(layers=n_head_layers,dropout_prob=dropout_p,head_type='classification', num_labels=1, activation_function=head_act_function,)
    model.add_prediction_head_from_config('AdapterBERT_head_adapter', config, overwrite_ok=True)
    
    #model.bert.prompt_tuning = nn.Identity()
    #adapter_path = '/ibmm_data/TemBERTure/model/BERT_regr/2_LAYER_HEAD/0-60Tm_only/RANDOM_IND_060ONLY_lr1e-3_headdrop0.3_replica1/output/checkpoint-57100/'
    #model.load_adapter(adapter_path+'AdapterBERT_adapter',with_head=True)
    #model.load_adapter(adapter_path+'AdapterBERT_adapter')
    #model.load_head(adapter_path + 'AdapterBERT_head_adapter')
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    model.active_head == 'AdapterBERT_head_adapter'
    model.train_adapter(["AdapterBERT_adapter"])
    model.set_active_adapters(["AdapterBERT_adapter"])
    print(model)
    logger.info(f' * USING HEAD DROPOUT: {dropout_p}')
    return model 


def AdapterBERT_cls(model_name,dropout_p=0.1,n_head_layers=2,head_act_function='tanh'):
    model = BertAdapterModel.from_pretrained(model_name) 
    logger.info('YES ADAPTERS MODEL')
    model.add_adapter("AdapterBERT_adapter",set_active=True)
    #model.add_classification_head('AdapterBERT_head_adapter',num_labels=1, layers=n_healayers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_functiond_layers,activation_function=head_act_function, dropout = dropout_p)
    #model.register_custom_head("AdapterBERT_head_adapter", CustomHead)
    #model.add_custom_head(head_type="AdapterBERT_head_adapter", head_name="AdapterBERT_head_adapter",num_labels=1, layers=n_head_layers,activation_function=head_act_function, dropout = dropout_p)
    config = dict(layers=n_head_layers,dropout_prob=dropout_p,head_type='classification', num_labels=1, activation_function=head_act_function,)
    model.add_prediction_head_from_config('AdapterBERT_head_adapter', config, overwrite_ok=True)
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
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
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path}')
    model = model.to('cuda')
    return model 


def AdapterBERT_sequential(model_name,adapter_path,dropout_p=0.4,n_head_layers=2,head_act_function='relu'):
    model = BertAdapterModel.from_pretrained(model_name) 
    model.load_adapter(adapter_path+'AdapterBERT_adapter',with_head=True)
    #model.add_classification_head('AdapterBERT_head_adapter',layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    # you have to add a new head and not use the weights of the pretrained cls head because the head config (i.e. activation functions is different)
    model.add_custom_head(head_type="AdapterBERT_head_adapter", head_name="AdapterBERT_head_adapter",layers=n_head_layers,dropout=dropout_p, num_labels=1, activation_function=head_act_function)
    model.active_head == 'AdapterBERT_head_adapter' #head not pretrained for cls task adapter!!
    model.train_adapter(["AdapterBERT_adapter"])
    model.delete_head('default')
    model.bert.prompt_tuning = nn.Identity()
    logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path} WITHOUT PRETRAINED HEAD, while adding new head config ')
    return model 
    

'''
from torch import nn
from torch.nn import  CrossEntropyLoss, MSELoss, Dropout
from modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
)

from modeling import Activation_Function_Class

# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        dropout_prob = self.config.get("dropout_prob", model_config.hidden_dropout_prob) #edited by me
        bias = self.config.get("bias", True)
        for l_id in range(self.config["layers"]):
            if l_id < self.config["layers"] - 1:
                if self.config["dropout"]: #edited by me
                    dropout = self.config.get("dropout",self.config["dropout"])
                    pred_head.append(nn.Dropout(dropout)) # edited by me
                if dropout_prob > 0:
                    pred_head.append(nn.Dropout(dropout_prob))
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                if self.config["activation_function"]:
                    pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if self.config["dropout"]: #edited by me
                    dropout = self.config.get("dropout",self.config["dropout"])
                    pred_head.append(nn.Dropout(dropout)) # edited by me
                if dropout_prob > 0:
                    pred_head.append(nn.Dropout(dropout_prob))
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"], bias=bias))
                elif "num_choices" in self.config:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1, bias=bias))
                else:
                    pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=bias))
                    if self.config["activation_function"]:
                        pred_head.append(Activation_Function_Class(self.config["activation_function"]))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        return None  # override for heads with output embeddings

    def get_label_names(self):
        return ["labels"]



class CustomHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        dropout=0.2, #edited 
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "dropout" : dropout, #edited 
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs '''
