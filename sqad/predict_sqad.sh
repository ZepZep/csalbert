#! /bin/bash

# Script for evaluation of fine-tuned ALBERT SQ(u)AD model
# Evaluates all current checkpoints, saves the best and deletes the rest of them, creates prediction files
# if best checkpoint (model.ckpt-best) is already present, only creates prediction files
# usage ./predict_sqad.sh csbase3 sqad_short_clean v2

albert="$1"
ds="$2"
spm="$3"

fds="dataset/${ds}_dev.json"
model_path="pretrained/${albert}_ckpt"

# set visible GPU, set to high value to use processor instead (very slow)
export CUDA_VISIBLE_DEVICES="0"
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
    --predict_batch_size=8 \
    --do_lower_case=True\
    --max_seq_length=512 \
    --doc_stride=256 \
    --max_query_length=64 \
    --warmup_proportion=.1 \
    --iterations_per_loop=100 \
    --n_best_size=10 \
    --max_answer_length=128 \
    --dropout_prob=0
