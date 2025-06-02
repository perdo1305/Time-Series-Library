#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # Use only GPU 0 for training
export PYTHONPATH=$(pwd)

# Set wandb project name
wandb_project="reduced_dataset_optimization_1_10"
# Uncomment and set your wandb entity if needed (username or team name)
# wandb_entity="your-username-or-team"

python run_optuna_wandb_2.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id battery_cycle \
    --model TimesNet \
    --data custom \
    --root_path ./CS2_35/ \
    --data_path /home/labrob/Documents/GIT/second_attempt_timesnet/shuffle_cycles/cycle-selection/selected_cycles_realistic_cycles_distributed_subset_10th.json \
    --features MS \
    --target "Capacity_Difference" \
    --freq t \
    --enc_in 4 \
    --dec_in 4 \
    --c_out 1 \
    --seq_len 48 \
    --label_len 24 \
    --pred_len 24 \
    --loss MSE \
    --train_epochs 20 \
    --patience 5 \
    --use_optuna \
    --optuna_study_name reduced_dataset_optimization_1_10 \
    --optuna_n_trials 50 \
    --use_wandb \
    --wandb_project ${wandb_project} \
    $(if [ ! -z "$wandb_entity" ]; then echo "--wandb_entity ${wandb_entity}"; fi)