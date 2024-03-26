# ToCoTR
This repository contains the source code for our paper **Typos Correction Training Against Misspellings from Text-to-Text Transformers**.

![ToCoTR model architecture](assets/ToCoTR.png)


## Installation
Our code is developed based on [CharacterBERT-DR](https://github.com/ielab/CharacterBERT-DR/).

First clone this repository, and then install with pip: `pip install --editable .`

> Note: The current code base has been tested with, `torch==1.8.1`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`, `textattack=0.3.4`


## Train

### Typos correction training
```
python3 -m tevatron.driver.train_typo \
        --model_name_or_path t5-base \
        --output_dir model_msmarco_typos_correction_training \
        --save_steps 40000 \
        --train_file_path msmarco_passage/corpus/corpus.jsonl \
        --per_device_train_batch_size 256 \
        --learning_rate 3e-4 \
        --max_steps 125763 \
        --dataloader_num_workers 10 \
        --max_len 32 \
        --logging_steps 100 \
        --model_type t5 \
        --typo_augmentation True \
        --typo_example_rate 0.8 \
        --typo_enhance True \
        --typo_rate 0.2 \
        --warmup_ratio 0.1
```

### Retriever training
```
python3 -m tevatron.driver.train \
        --model_name_or_path model_msmarco_typos_correction_training/checkpoint-final \
        --output_dir model_msmarco_retrieval \
        --passage_field_separator [SEP] \
        --save_steps 160000 \
        --dataset_name Tevatron/msmarco-passage \
        --train_dir msmarco_passage/train \
        --per_device_train_batch_size 8 \
        --learning_rate 5e-5 \
        --max_steps 150000 \
        --dataloader_num_workers 10 \
        --logging_steps 500 \
        --model_type t5 \
        --self_teaching True
```

Our trained model checkpoints for you to download: (ToCoTR)(https://drive.google.com/file/d/1cIr0JvkoqPLsme_zWIDMlr7Ufw1Smo6w/view?usp=sharing)

In this training phase, you can optimize dense retrievers with different objectives by setting different combinations of parameters such as `--contrastive_training`, `--typo_augmentation`, `--self_teaching`, `--rodr_training`.

## Inference

### Encode queries and corpus
After you have the trained model, you can run the following command to encode queries and corpus into dense vectors:

```
mkdir msmarco_tocotr_st_embs
# encode query
python3 -m tevatron.driver.encode \
  --output_dir temp \
  --model_name_or_path model_msmarco_retrieval/checkpoint-final \
  --per_device_eval_batch_size 256 \
  --encode_in_path marco_dev/queries.dev.small.tsv \
  --encoded_save_path msmarco_tocotr_st_embs/query_marco_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --character_query_encoder False \
  --model_type t5

# encode corpus
for s in $(seq -f "%02g" 0 19)
do
    python3 -m tevatron.driver.encode \
        --output_dir temp \
        --model_name_or_path model_msmarco_retrieval/checkpoint-final \
        --per_device_eval_batch_size 256 \
        --p_max_len 128 \
        --dataset_name Tevatron/msmarco-passage-corpus \
        --encode_in_path msmarco_passage/corpus/corpus.jsonl \
        --encoded_save_path msmarco_tocotr_st_embs/corpus_emb.${s}.pkl \
        --encode_num_shard 20 \
        --encode_shard_index ${s} \
        --character_query_encoder False \
        --model_type t5 \
        --passage_field_separator [SEP]
done
```

### Retrieval
Run the following commands to generate ranking file and convert it to TREC format:

```
python3 -m tevatron.faiss_retriever \
    --query_reps msmarco_tocotr_st_embs/query_marco_emb.pkl \
    --passage_reps msmarco_tocotr_st_embs/'corpus_emb.*.pkl' \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to dr_tocotr_marco_rank.txt

python3 -m tevatron.utils.format.convert_result_to_trec \
              --input dr_tocotr_marco_rank.txt \
              --output dr_tocotr_marco_rank.txt.trec
```
