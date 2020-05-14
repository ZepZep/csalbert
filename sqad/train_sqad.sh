#! /bin/bash

# ./train_sqad.sh csbase3 v2 sqad_short_clean

albert="$1"
spm="$2"
ds="$3"

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
    --max_seq_length=256 \
    --doc_stride=128 \
    --max_query_length=64 \
    --train_batch_size=16 \
    --learning_rate=5e-6 \
    --num_train_epochs=30 \
    --warmup_proportion=.1 \
    --iterations_per_loop=50 \
    --save_checkpoints_steps=500 \
    --n_best_size=10 \
    --max_answer_length=128 \
    --dropout_prob=0
    
#     --predict_file=dataset/sqad_dev.json \
#     --predict_batch_size=8 \
#     --do_train \
#     --train_file=dataset/sqad_train.json \
#     --init_checkpoint=pretrained/csbase1_ckpt/model.ckpt-best \
