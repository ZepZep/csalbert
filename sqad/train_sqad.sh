#! /bin/bash

# Script for fine-tuning ALBERT on SQ(u)AD
# usage ./train_sqad.sh csbase3 sqad_short_clean v2

albert="$1"     # name of ALBERT model
ds="$2"         # dataset name
spm="$3"        # additional feature file identifier, useful if using multiple spm models

fds="dataset/${ds}_train.json"
model_path="pretrained/${albert}_ckpt"

python albert/run_squad_v2.py \
    --albert_config_file=$model_path/albert_config.json \
    --spm_model_file=$model_path/spm.model \
    --output_dir="models/${albert}_$ds" \
    --do_train=True \
    --train_file=$fds \
    --train_feature_file="features/${ds}_train_ff$spm.tf" \
    --do_lower_case=True\
    --max_seq_length=512 \
    --doc_stride=256 \
    --max_query_length=64 \
    --train_batch_size=8 \
    --learning_rate=5e-6 \
    --num_train_epochs=15 \
    --warmup_proportion=.1 \
    --iterations_per_loop=50 \
    --save_checkpoints_steps=500 \
    --n_best_size=10 \
    --max_answer_length=32 \
    --dropout_prob=0
