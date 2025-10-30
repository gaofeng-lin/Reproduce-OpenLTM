export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
# token_num=7
# token_len=168

input_len=672
patch_size=96

token_len=$patch_size
token_num=$((input_len/patch_size))

# seq_len=$[$token_num*$token_len]
seq_len=$input_len
# seq_len=$token_len

python -u run.py \
  --task_name changecode_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last


# for test_pred_len in 96
# do
# python -u run.py \
#   --task_name forecast \
#   --is_training 0 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1 \
#   --model $model_name \
#   --data UnivariateDatasetBenchmark  \
#   --seq_len $seq_len \
#   --input_token_len $token_len \
#   --output_token_len $token_len \
#   --test_seq_len $seq_len \
#   --test_pred_len 96 \
#   --batch_size 256 \
#   --learning_rate 0.0005 \
#   --train_epochs 10 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --gpu 0 \
#   --lradj type1 \
#   --use_norm \
#   --e_layers 1 \
#   --valid_last \
#   --test_dir changecode_forecast_ETTh1_timer_xl_UnivariateDatasetBenchmark_sl2880_it2880_ot2880_lr0.0005_bt256_wd0_el1_dm512_dff2048_nh8_cosFalse_test_0
# done

  # --data UnivariateDatasetBenchmark  \
  # --data MultivariateDatasetBenchmark  \
    # --patch_size 96 \
# --nonautoregressive \