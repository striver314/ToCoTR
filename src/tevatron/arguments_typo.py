from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_type: str = field(default="t5", metadata={"help": "One of 't5', 'bart'"})


@dataclass
class DataArguments:
    train_file_path: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    corpus_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )

    typo_augmentation: bool = field(default=True)
    typo_rate: float = field(default=0.15)
    typo_example_rate: float = field(default=0.9)
    typo_enhance: bool = field(default=True)
    
    max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
