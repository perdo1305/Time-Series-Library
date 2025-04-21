# Set to use GPU 0 
export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./CS2_35 \
  --data_path combined_data.csv \
  --model_id cs2_forecast \
  --model $model_name \
  --data cs2 \
  --features M \
  --target "Voltage(V)" \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 18 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'mse' \
  --seasonal_patterns 'Custom'