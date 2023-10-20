import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.system('ulimit -n 10000')

import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument("--do_train", help="", type=bool,default=False)
parser.add_argument("--do_test", help="", type=bool,default=False)
parser.add_argument("--do_inference", help="", type=bool,default=False)

parser.add_argument("--model_name_or_path", help="Pre-trained model to use for training or best model path for inference/test", type=str,required=False) #"ElnaggarLab/ankh-base"
parser.add_argument("--model_type", help="T5, Bert etc", type=str,required=False) 
parser.add_argument("--with_adapters", help="", type=bool,default=None)

parser.add_argument("--cls_train", help="", type=str,required=False)
parser.add_argument("--cls_val", help="", type=str,required=False)
parser.add_argument("--regr_train", help="", type=str,required=False)
parser.add_argument("--regr_val", help="", type=str,required=False)


parser.add_argument("--wandb_project", help="", type=str,required=False,default='./test')
parser.add_argument("--wandb_run_name", help="", type=str,required=False,default=None)

parser.add_argument("--lr", help="", type=float,required=False,default=1e-5)
parser.add_argument("--weight_decay", help="", type=float,required=False,default=0.0)
#parser.add_argument("--scheduler", help="", type=str,required=False,default = 'linear')
parser.add_argument("--warmup_ratio", help="", type=float,required=False, default= 0)
parser.add_argument("--head_dropout", help="", type=float,required=False, default= 0.1)
parser.add_argument("--per_device_train_batch_size", help="", type=int,required=False, default= 16)
parser.add_argument("--per_device_eval_batch_size", help="", type=int,required=False, default= 16)

# if SequentialBERT & test
parser.add_argument("--best_model_path", help="", type=str,required=False,default=None)

# if test
parser.add_argument("--test_data", help="", type=str,required=False,default=None)
parser.add_argument("--task", help="", type=str,required=False,default=None)
parser.add_argument("--test_out_path", help="", type=str,required=False,default='./test')



args = parser.parse_args()

LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
#SCHEDULER = args.scheduler 
WARMUP_RATIO = args.warmup_ratio
DROPOUT = args.head_dropout
per_device_train_batch_size = args.per_device_train_batch_size
per_device_eval_batch_size = args.per_device_eval_batch_size



if __name__ == "__main__":
    import torch

    is_cuda = torch.cuda.is_available()
    print(is_cuda)
    
    if args.do_train:
        from train import Train
        os.makedirs('./data', exist_ok=True)
        print(args.with_adapters)
        Train(model_name = args.model_name_or_path, 
                        model_type = args.model_type,
                        adapters = args.with_adapters,
                        cls_train = args.cls_train,
                        cls_val = args.cls_val,
                        regr_train = args.regr_train,
                        regr_val = args.regr_val,
                        wandb_project = args.wandb_project,
                        wandb_run_name = args.wandb_run_name,
                        adapter_path=args.best_model_path)
    
    if args.do_test:
        BATCH_SIZE=16
        
        os.makedirs('./test', exist_ok=True)
        os.chdir('./test/')
        
        from model import model_init
        
        tokenizer,model = model_init(model_type='BERTSequential', model_name= args.model_name_or_path, adapters=args.with_adapters,adapter_path=args.best_model_path)
        
        from inference import test_out
        test_out(args.task, model=model, tokenizer=tokenizer, raw_test_df=args.test_data, best_model_path = args.best_model_path, BATCH_SIZE=BATCH_SIZE)

            
        
        
        
