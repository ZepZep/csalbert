#! /bin/bash

python -m albert.run_squad_v2 \
    --albert_config_file=pretrained/csbase1_ckpt/albert_config.json \
    --output_dir= models/cspase1_sqad \
    --train_file=dataset/sqad_train.json \
    --init_checkpoint=pretrained/csbase1_ckpt/ \
    --spm_model_file=pretrained/spm/tenten_smp_5_30K.model \
    --do_lower_case=True\
    --max_seq_length=256 \
    --doc_stride=200 \
    --max_query_length=64 \
    --do_train \
    --train_batch_size=16 \
    --learning_rate=5e-5 \
    --num_train_epochs=2.0 \
    --warmup_proportion=.1 \
    --iterations_per_loop=100 \
    --save_checkpoints_steps=250 \
    --n_best_size=20 \
    --max_answer_length=128 \
    --dropout_prob=0
    
#     --predict_file=dataset/sqad_dev.json \
#     --predict_batch_size=8 \
