import logging
import os
import sys

import datasets
from transformers import AutoConfig, T5Tokenizer, BartTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments_typo import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import QCTrainDataset, QPCollator
from tevatron.preprocessor import HFCorpusPreProcessor
from tevatron.modeling import QCDenseModel
from tevatron.trainer import QCDenseTrainer as QCTrainer, GCTrainer

logger = logging.getLogger(__name__)

MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        
    assert model_args.model_type in list(MODEL_TYPE_TO_TOKENIZER.keys()), "model type should be 't5' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.warning("Training/evaluation parameters %s", training_args)
    logger.warning("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    # set seed
    set_seed(training_args.seed)

    # Load pretrain model and toknizer
    num_labels = 1
    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    
    model = QCDenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    data_collator = QPCollator(
        tokenizer,
        max_p_len=data_args.max_len,
        max_q_len=data_args.max_len
    )

    # Get dataset
    if data_args.train_file_path is not None:
        train_dataset = datasets.load_dataset('json',
                                              data_files=data_args.train_file_path,
                                              cache_dir=model_args.cache_dir)["train"]
    else:
        raise ValueError("Train File can not be ignored.")
        
    # map: token -> input_ids
    train_dataset = train_dataset.map(
        HFCorpusPreProcessor(tokenizer, data_args.max_len, separator=data_args.corpus_separator),
        batched=False,
        num_proc=data_args.dataset_proc_num,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on typo-correction training dataset",
    )
    train_dataset = QCTrainDataset(data_args,
                                 train_dataset,
                                 tokenizer,
                                 cache_dir=model_args.cache_dir)

    if training_args.grad_cache:
        trainer_cls = GCTrainer
    else:
        trainer_cls = QCTrainer

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, f"checkpoint-final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, f"checkpoint-final"))


if __name__ == "__main__":
    main()
