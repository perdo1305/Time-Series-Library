# Set to use GPU 0 (your NVIDIA GeForce RTX 3080)
export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 18 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'