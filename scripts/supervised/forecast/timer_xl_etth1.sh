export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
# token_num=7
# token_len=168

# seq_len表示输入序列的总长度
# 这里我们设置为24，即使用过去24个时间步的数据进行预测
seq_len=672
output_len=672
# input_token_len表示每个token的长度，因为模型会对序列进行分块处理
# 这个值应该对应论文的patch_size，但是代码中并没有使用patch_size参数
input_token_len=96
# output_token_len通常与input_token_len相同。模型源码部分的注释中也是让输入和输出序列长度相等
output_token_len=$input_token_len


python -u run.py \
  --task_name changecode_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --output_len $output_len \
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
  --nonautoregressive \
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