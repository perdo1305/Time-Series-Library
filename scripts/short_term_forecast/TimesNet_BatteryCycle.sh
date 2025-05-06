#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./CS2_35/\
  --data_path combined_data.csv \
  --model_id battery_cycle_prediction_combined_data \
  --model $model_name \
  --data custom \
  --features MS \
  --e_layers 1 \
  --d_layers 1 \
  --factor 2 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --batch_size 8 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'BatteryCycle_combined_data' \
  --itr 1 \
  --learning_rate 0.001 \
  --target 'Voltage(V)' \
  --freq 'h' \
  --loss 'MSE'