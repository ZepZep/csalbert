#! /bin/bash

# ./predict_sqad.sh csbase3 v2 sqad_short_clean

albert="$1"
spm="$2"
ds="$3"

fds="dataset/${ds}_dev.json"
model_path="pretrained/${albert}_ckpt"


export CUDA_VISIBLE_DEVICES="1"
export TF_CPP_MIN_LOG_LEVEL="3"

python albert/run_squad_v2.py \
    --albert_config_file=$model_path/albert_config.json \
    --spm_model_file=$model_path/spm.model \
    --output_dir="models/${albert}_$ds" \
    --do_predict=True \
    --train_file=$fds \
    --predict_file=$fds \
    --predict_feature_file="features/${ds}_predict_ff$spm.tf" \
    --predict_feature_left_file="features/${ds}_predict_left_ff$spm.tf" \
    --predict_batch_size=16 \
    --do_lower_case=True\
    --max_seq_length=256 \
    --doc_stride=200 \
    --max_query_length=64 \
    --warmup_proportion=.1 \
    --iterations_per_loop=100 \
    --n_best_size=10 \
    --max_answer_length=128 \
    --dropout_prob=0
    
#     --predict_file=dataset/sqad_dev.json \
#     --predict_batch_size=8 \
#     --do_train \
#     --train_file=dataset/sqad_train.json \
#     --init_checkpoint=pretrained/csbase1_ckpt/model.ckpt-best \
