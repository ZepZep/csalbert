#! /bin/bash

# Train SQAD for BERT models
# requires the BERT repository in directory gbert
# usage: ./bert_train_sqad.sh csbase3 sqad_extract 0

bert="$1"
ds="$2"
gpu="$3"

fds="dataset/${ds}_train.json"
model_path="pretrained/${bert}"

export CUDA_VISIBLE_DEVICES="$gpu"

python gbert/run_squad.py \
  --vocab_file="$model_path/vocab.txt" \
  --bert_config_file="$model_path/bert_config.json" \
  --init_checkpoint="$model_path/bert_model.ckpt" \
  --do_train=True \
  --train_file="$fds" \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=10 \
  --max_seq_length=512 \
  --doc_stride=256 \
  --output_dir="models/${bert}_$ds" \
  --version_2_with_negative=True \
  --save_checkpoints_steps=500 \
  --n_best_size=10 \
  --warmup_proportion=0.05 \
  --save_checkpoints_steps=500 \
  --max_answer_length=64
