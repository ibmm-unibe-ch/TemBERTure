import sys

from dataclasses import asdict, dataclass, field, fields
from multiprocessing import cpu_count
from torch.utils.data import Dataset

from dataclasses import asdict, dataclass, field, fields
from transformers import TrainingArguments


@dataclass
class ModelArgs (TrainingArguments):

    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    #max_grad_norm: float = 1.0
    max_seq_length: int = 512
    n_gpu: int = 1
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = False
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = True
    save_optimizer_and_scheduler: bool = True
    use_early_stopping: bool = True
    


