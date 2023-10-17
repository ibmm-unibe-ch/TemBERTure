from transformers import TrainingArguments, Trainer, AdapterTrainer
from utils import logger, compute_metrics_for_regression, compute_metrics_for_classification
import torch
from data_prep import TrainerData, ClsData, RegrData, TrainerDataT5
    


# trainer specific params
from main import LEARNING_RATE,WEIGHT_DECAY,WARMUP_RATIO, per_device_train_batch_size, per_device_eval_batch_size
learning_rate = LEARNING_RATE
weight_decay = WEIGHT_DECAY
#scheduler = SCHEDULER
warmup_ratio = WARMUP_RATIO
train_batch_size = per_device_train_batch_size
eval_batch_size = per_device_eval_batch_size


logger.info(' ** TRAINER PARAMS **')
logger.info(f'LEARNING RATE USED {learning_rate}')
logger.info(f'WEIGHT DECAY USED {weight_decay}')
#logger.info(f'SCHEDULER USED {scheduler}')
logger.info(f'WARMUP RATIO USED {warmup_ratio}')
logger.info(f'TRAIN-EVAL BATXH SIZES USED {per_device_train_batch_size,per_device_eval_batch_size}')

    
def BERTTrainer(cls_train,cls_val,regr_train,regr_val,tokenizer,model,model_type, adapters):
    print(f' ** Running {model_type} model&task, with adapters set to {adapters} **')
    logger.info(f' ** Running {model_type} model&task, with adapters set to {adapters} ** ')
    
    # task specific data
    if model_type == 'BERT_cls':
        train_df, eval_df = ClsData(cls_train, cls_val)
    elif model_type == 'BERT_regr' or model_type == 'BERTSequential':
        train_df, eval_df = RegrData(regr_train, regr_val)
    
     
    if adapters:
        per_device_train_batch_size = train_batch_size
        per_device_eval_batch_size = eval_batch_size
    else:
        per_device_train_batch_size=  4 #if used the model without adapters CUDA OUT OF MEMORY
        per_device_eval_batch_size = 4

    
    
    train_df["input_text"] = [" ".join("".join(sample.split())) for sample in train_df["input_text"]] #as required by rostlab models by their tokeizer
    eval_df["input_text"] = [" ".join("".join(sample.split())) for sample in eval_df["input_text"]] #as required by rostlab models by their tokeizer

    logger.info('Datasets input_text edited preview as required by rostlab tokenizer (blank space between aa):')
    logger.info(train_df[:2])
    logger.info(eval_df[:2])


    train_df, eval_df = TrainerData(train_df, eval_df,tokenizer)

    from model_args import ModelArgs
    training_args = ModelArgs(
        output_dir="./output",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=10,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        remove_unused_columns=False,
        push_to_hub=False,
        learning_rate = learning_rate, 
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        weight_decay = weight_decay,
        #lr_scheduler_type = scheduler,
        warmup_ratio = warmup_ratio,
        report_to='wandb',
        save_on_each_node=True,
        greater_is_better=False,
        seed = 42,
        max_seq_length = 512,
        use_early_stopping = True,
        
    )
    
    logger.info (training_args)
    
    # Definisci un dizionario che associa il tipo di modello ai trainer appropriati
    trainer_mapping = {
        'BERT_regr': RegressionAdapterTrainer if adapters else RegressionTrainer,
        'BERT_cls': ClassificationAdapterTrainer if adapters else ClassificationTrainer,
        'BERTSequential': RegressionAdapterTrainer if adapters else RegressionTrainer,
    }

    # Ottieni il trainer corretto in base al tipo di modello
    trainer_class = trainer_mapping.get(model_type)
    
    logger.info(f'TRAINER CLASS {trainer_class}')

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=eval_df,
        tokenizer=tokenizer,
        compute_metrics=(
            compute_metrics_for_regression
            if model_type == 'BERT_regr' or model_type == 'BERTSequential'
            else compute_metrics_for_classification ) )
    
            
    return trainer


def T5Trainer(cls_train,cls_val,regr_train,regr_val,tokenizer,model):
    print('Running T5 multi task training (binary classification and regression)')
    logger.info('Running T5 multi task training (binary classification and regression)')

    from t5_utils import T5Data 
    train_df, eval_df =  T5Data(cls_train,cls_val,regr_train,regr_val)
    
    train_df, eval_df = TrainerDataT5(train_df, eval_df,tokenizer)
    
    from model_args import T5Args
    from transformers import GenerationConfig
    
    logger.info(f'LEARNING RATE USED {learning_rate}') 
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

    generation_config = GenerationConfig(
    num_beams=1,
    early_stopping=True,
    decoder_start_token_id=model.config.decoder_start_token_id,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,

    )
    training_args = Seq2SeqTrainingArguments(
        # f"{model_name}-finetuned-xsum",
        output_dir="./output",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        #save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        remove_unused_columns=False,
        push_to_hub=False,
        learning_rate = learning_rate, #LEARNING_RATE,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        #weight_decay = WEIGHT_DECAY,
        report_to='wandb',
        save_on_each_node=True,
        greater_is_better=False,
        seed = 42,
        #max_seq_length = 512,
        predict_with_generate = True,
        #length_penalty = 2.0,
        #generation_max_length = 5,
        #max_steps  = -1,
        #generation_num_beams = 1,
        #num_return_sequences  = 1,
        #preprocess_inputs  = True,
        #repetition_penalty = 1.0,
        lr_scheduler_type = "constant",
        #adafactor_relative_step = False,
        #adafactor_scale_parameter = False,
        #adafactor_warmup_init = False,
        #optimizer = "Adafactor",
        adafactor = True,
        #top_k = None,
        #top_p = None,
        #use_multiprocessed_decoding = True,
        #adafactor_beta1  = None,
        #adafactor_clip_threshold  = 1.0,
        #adafactor_decay_rate  = -0.8,
        #adafactor_eps = field(default_factory=lambda: (1e-30, 1e-3))
        #adam_betas= field(default_factory=lambda: (0.9, 0.999))
        #adam_epsilon = 1e-8,
        fp16 = False,
        #generation_config = generation_config,

        )
    
    from utils import prep_compute_metrics_for_multitaskT5
    
    compute_metrics_for_multitaskT5 = prep_compute_metrics_for_multitaskT5(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_df,
        #data_collator=data_collator,
        eval_dataset = eval_df,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics_for_multitaskT5
    )
    
    logger.info(training_args)
    
    return trainer


class RegressionAdapterTrainer(AdapterTrainer): 
    '''
    1.Extract the "labels" from the inputs dictionary using the pop method. This suggests that the input dictionary contains a key named "labels" that corresponds to the ground truth labels for the regression task.
    2.Pass the remaining inputs dictionary to the model to obtain the model's outputs.
    3.Extract the logits from the outputs tensor. It assumes that the model's output is a tensor, and it retrieves the logits corresponding to the first element of each sample ([:, 0]).
    4.Compute the mean squared error (MSE) loss between the logits and the labels using the torch.nn.functional.mse_loss function. This calculates the squared difference between the predicted values and the ground truth labels.
    The method returns either the computed loss alone (loss) or a tuple containing the loss and the model's outputs ((loss, outputs)) based on the return_outputs flag.'''

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)
        
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        #print('logits',logits)# Assuming your model returns logits directly
        #print('labels', labels)
        #print('logits',logits.size())
        #print('labels', labels.size())

        #logits = outputs.logits.squeeze(dim=-1) 
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
class RegressionTrainer(Trainer): 
    '''
    1.Extract the "labels" from the inputs dictionary using the pop method. This suggests that the input dictionary contains a key named "labels" that corresponds to the ground truth labels for the regression task.
    2.Pass the remaining inputs dictionary to the model to obtain the model's outputs.
    3.Extract the logits from the outputs tensor. It assumes that the model's output is a tensor, and it retrieves the logits corresponding to the first element of each sample ([:, 0]).
    4.Compute the mean squared error (MSE) loss between the logits and the labels using the torch.nn.functional.mse_loss function. This calculates the squared difference between the predicted values and the ground truth labels.
    The method returns either the computed loss alone (loss) or a tuple containing the loss and the model's outputs ((loss, outputs)) based on the return_outputs flag.'''

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        #print(labels.size())
        print(labels)
        
        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)
        
        outputs = model(**inputs)
        
        #print(outputs)
        #print(labels)
        #logits = outputs[0][:, 0]
        logits = outputs #per la classe senza adater che la sua PreTrainedModel class (BERT_regr) va lasciato con 1 dim
        logits = logits.view(labels.size()) #labels and logits were not in the same size, before torch.Size([4]) torch.Size([4, 1]) now torch.Size([4])

        #print(logits.size())
        #print(logits)
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ClassificationAdapterTrainer(AdapterTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)

        outputs = model(**inputs)
        logits = outputs[0]
        #logits = outputs[0][:, 0]
        #print('logits',logits)# Assuming your model returns logits directly
        #print('labels', labels)
        #print('logits',logits.size())
        #print('labels', labels.size())

        # Ensure labels and logits have the correct shape
        labels = labels.view(-1, 1).float()  # Reshape labels to [batch_size, 1]
        
        
        logits = torch.sigmoid(logits)
        #print(labels)
        #print('logits1',logits)
        #print('new')
        #print('labels', labels)
        #print('logits',logits.size())
        #print('labels', labels.size())
        loss = torch.nn.functional.binary_cross_entropy(logits, labels)

        # Calculate binary cross-entropy loss
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        print("Is model on GPU?", model_on_gpu)

        outputs = model(**inputs)
        logits = outputs[0] # Assuming your model returns logits directly
        #logits = outputs[0][:, 0]
        #print('logits',logits)
        #print('labels', labels)

        # Ensure labels and logits have the correct shape
        labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
        
        # Calculate binary cross-entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss
