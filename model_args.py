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
    


@dataclass
class T5Args(ModelArgs):
    """
    Model args for a T5Model
    """

    model_class: str = "T5Model"
    dataset_class: Dataset = None
    do_sample: bool = False
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    length_penalty: float = 2.0
    max_length: int = 512
    max_steps: int = -1
    num_beams: int = 1
    num_return_sequences: int = 1
    preprocess_inputs: bool = True
    repetition_penalty: float = 1.0
    lr_scheduler_type: str = "constant"
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = False
    adafactor_warmup_init: bool = False
    learning_rate: float =  1e-3 #3e-4 #
    optimizer: str = "Adafactor"
    adafactor: bool = True
    special_tokens_list: list = field(default_factory=list)
    top_k: float = None
    top_p: float = None
    use_multiprocessed_decoding: bool = True
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    adam_epsilon: float = 1e-8
    fp16: bool = False
    predict_with_generate : bool = True


