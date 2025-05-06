#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # Use only GPU 0 for training

# Select TimesNet as the model architecture
model_name=TimesNet

python run_optuna.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id battery_cycle \
    --model ${model_name} \
    --data custom \
    --root_path ./CS2_35/ \
    --data_path /home/labrob/Documents/GIT/second_attempt_timesnet/shuffle_cycles/cycle-selection/selected_cycles.json \
    --features MS \
    --target "Capacity_Difference" \
    --freq m \
    --enc_in 14 \
    --dec_in 14 \
    --c_out 14 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --loss MSE \
    --train_epochs 5 \
    --patience 3 \
    --use_optuna \
    --optuna_study_name battery_cycle_optimization \
    --optuna_n_trials 50