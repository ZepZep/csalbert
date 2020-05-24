#! /bin/bash

# Predict SQAD for BERT models
# requires the BERT repository in directory gbert
# usage: ./bert_predict_sqad.sh csbase3 sqad_extract

bert="$1"
ds="$2"

fds="dataset/${ds}_dev.json"
model_path="pretrained/${bert}"

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="3"

python gbert/run_squad.py \
  --vocab_file="$model_path/vocab.txt" \
  --bert_config_file="$model_path/bert_config.json" \
  --init_checkpoint="$model_path/bert_model.ckpt" \
  --do_predict=True \
  --predict_file="$fds" \
  --predict_batch_size=8 \
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
