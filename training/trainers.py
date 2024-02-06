from transformers import TrainingArguments, Trainer
from adapters import AdapterTrainer
from utils import logger, compute_metrics_for_regression, compute_metrics_for_classification
import torch
from data_prep import TrainerData, ClsData, RegrData
    


# trainer specific params
from main import LEARNING_RATE,WEIGHT_DECAY,WARMUP_RATIO, per_device_train_batch_size, per_device_eval_batch_size
learning_rate = LEARNING_RATE
weight_decay = WEIGHT_DECAY
warmup_ratio = WARMUP_RATIO
train_batch_size = per_device_train_batch_size
eval_batch_size = per_device_eval_batch_size


logger.info(' ** TRAINER PARAMS **')
logger.info(f'LEARNING RATE USED {learning_rate}')
logger.info(f'WEIGHT DECAY USED {weight_decay}')
logger.info(f'WARMUP RATIO USED {warmup_ratio}')
logger.info(f'TRAIN-EVAL BATXH SIZES USED {per_device_train_batch_size,per_device_eval_batch_size}')


    
def BERTTrainer(cls_train,cls_val,regr_train,regr_val,tokenizer,model,model_type, adapters):
    print(f' ** Running {model_type} model&task, with adapters set to {adapters} **')
    logger.info(f' ** Running {model_type} model&task, with adapters set to {adapters} ** ')
    
    # task specific data
    if model_type == 'BERT_cls':
        train_df, eval_df = ClsData(cls_train, cls_val)
    elif model_type == 'BERT_regr' or model_type == 'BERTSequential' or "BERTInference":
        train_df, eval_df = RegrData(regr_train, regr_val)
    
     
    if adapters:
        per_device_train_batch_size = train_batch_size
        per_device_eval_batch_size = eval_batch_size
    else:
        per_device_train_batch_size =  4 #if used the model without adapters CUDA OUT OF MEMORY
        per_device_eval_batch_size = 4

    
    train_df["input_text"] = [" ".join("".join(sample.split())) for sample in train_df["input_text"]] #as required by rostlab models by their tokeizer
    eval_df["input_text"] = [" ".join("".join(sample.split())) for sample in eval_df["input_text"]] #as required by rostlab models by their tokeizer

    logger.info('Datasets input_text edited preview as required by rostlab tokenizer (blank space between aa):')
    logger.info(train_df[:2])
    logger.info(eval_df[:2])

    train_df, eval_df = TrainerData(train_df, eval_df,tokenizer)
    
    import random
    random.seed(10)
    print(random.random()) 

    

    from model_args import ModelArgs
    training_args = ModelArgs(
        output_dir="./output",
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        #num_train_epochs=1,
        #evaluation_strategy="epoch",
        #save_strategy='steps',
        #save_steps=7750,
        #save_total_limit=10,
        num_train_epochs = 50,
        evaluation_strategy = "epoch",
        save_strategy = 'epoch',
        remove_unused_columns = False,
        push_to_hub = False,
        learning_rate = learning_rate, 
        metric_for_best_model = "loss",
        load_best_model_at_end = True,
        weight_decay = weight_decay,
        #lr_scheduler_type = scheduler,
        warmup_ratio = warmup_ratio,
        report_to='wandb',
        save_on_each_node = True,
        greater_is_better = False,
        seed = 42,
        max_seq_length = 512,
        use_early_stopping = True,
        
    )
    
    logger.info(f'TRAINER ARGS {training_args}')
    logger.info (training_args)
    
    # Definisci un dizionario che associa il tipo di modello ai trainer appropriati
    trainer_mapping = {
        'BERT_regr': RegressionAdapterTrainer if adapters else RegressionTrainer,
        'BERT_cls': ClassificationAdapterTrainer if adapters else ClassificationTrainer,
        'BERTSequential': RegressionAdapterTrainer if adapters else RegressionTrainer,
        'BERTInference': RegressionAdapterTrainer if adapters else RegressionTrainer,
    }

    # Ottieni il trainer corretto in base al tipo di modello
    trainer_class = trainer_mapping.get(model_type)
    
    logger.info(f'TRAINER CLASS {trainer_class}')

    trainer = trainer_class(
        model=model.to('cuda'),
        args=training_args,
        train_dataset=train_df,
        eval_dataset=eval_df,
        tokenizer=tokenizer,
        compute_metrics=(
            compute_metrics_for_regression
            if model_type == 'BERT_regr' or model_type == 'BERTSequential' or model_type == "BERTInference"
            else compute_metrics_for_classification ) )
    
            
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
        labels = labels.to('cuda')
        
        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)
        
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
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
        
        #inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        #model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)
        
        outputs = model(**inputs)

        #logits = outputs[0][:, 0]
        logits = outputs #per la classe senza adater che la sua PreTrainedModel class (BERT_regr) va lasciato con 1 dim
        logits = logits.view(labels.size()) #labels and logits were not in the same size, before torch.Size([4]) torch.Size([4, 1]) now torch.Size([4])

        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ClassificationAdapterTrainer(AdapterTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        print("Are inputs on GPU?", inputs_on_gpu)
        model_on_gpu = next(model.parameters()).is_cuda
        print("Is model on GPU?", model_on_gpu)

        outputs = model(**inputs)
        logits = outputs[0]
        #logits = outputs[0][:, 0]

        # Ensure labels and logits have the correct shape
        labels = labels.view(-1, 1).float()  # Reshape labels to [batch_size, 1]
        
        # without logits so you have to pass thethe logits throught he sigmoid function
        logits = torch.sigmoid(logits)
        loss = torch.nn.functional.binary_cross_entropy(logits, labels)

        # Calculate binary cross-entropy loss
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        #inputs_on_gpu = all(tensor.is_cuda for tensor in inputs.values())
        #print("Are inputs on GPU?", inputs_on_gpu)
        #model_on_gpu = next(model.parameters()).is_cuda
        #print("Is model on GPU?", model_on_gpu)

        outputs = model(**inputs)
        logits = outputs[0] # Assuming your model returns logits directly
        #logits = outputs[0][:, 0]

        # Ensure labels and logits have the correct shape
        labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
        
        # Calculate binary cross-entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss
