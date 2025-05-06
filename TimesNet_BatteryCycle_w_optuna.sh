#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # Use only GPU 0 for training


model_name=TimesNet   # Select TimesNet as the model architecture

python run_optuna.py \
    --task_name long_term_forecast \   # Define the task type: forecasting future values
    --is_training 1 \                   # Enable training mode (1=training, 0=inference only)
    --model_id battery_cycle \          
    --model ${model_name} \             # Model architecture 
    --data custom \                     # Use custom dataset loader
    --root_path ./CS2_35/ \             # Base directory for data
    --data_path organized_battery_data_nice.csv \  # Specific data file to use
    --features MS \                     # Feature type: MS = Multiple variables, Single output
    --target "Capacity_Difference" \    # Target column to predict
    --freq m \                          # Data frequency (m = minutes)
    --enc_in 14 \                       # Number of input features (14 measurement variables)
    --dec_in 14 \                       # Number of input features for decoder
    --c_out 14 \                        # Number of output features
    --seq_len 48 \                      # Input sequence length (historical time steps)
    --label_len 24 \                    # Overlap between input and prediction (for attention)
    --pred_len 24 \                     # Prediction horizon (future time steps to predict)
    --loss MSE \                        # Loss function
    --train_epochs 50 \                 # Maximum number of training epochs
    --patience 3 \                      # Early stopping
    --use_optuna \                      # Enable Optuna 
    --optuna_study_name battery_cycle_optimization \  # Name of the Optuna 
    --optuna_n_trials 50                # Number of hyperparameter combinations to try