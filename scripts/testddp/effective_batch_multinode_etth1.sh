#!/bin/bash

# 每台机器使用显卡数目
nproc_per_node=2  # 根据实际情况调整
# 主机器ip
MASTER_ADDR=10.200.8.34
# 主机器端口号，可以随意，只要不冲突
MASTER_PORT=64223
# 机器编号，主机器必须为0
node_rank=$1
# 使用的机器数量
nnodes=$2
# 每个进程的线程数目
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0,1
model_name=timer_xl
token_num=7
token_len=672
# seq_len=$[$token_num*$token_len]
# 修复数组计算语法
seq_len=$((token_num * token_len))

# 分布式训练参数
DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --node_rank $node_rank --nnodes $nnodes --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# 使用torch.distributed.launch启动训练
# python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS run.py \
#   --task_name effectivebatchmultinodeforecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1 \
#   --model $model_name \
#   --data MultivariateDatasetBenchmark  \
#   --seq_len $seq_len \
#   --input_token_len $token_len \
#   --output_token_len $token_len \
#   --test_seq_len $seq_len \
#   --test_pred_len 96 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --train_epochs 10 \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --gpu 0 \
#   --devices 0,1 \
#   --ddp \
#   --lradj type1 \
#   --use_norm \
#   --e_layers 1 \
#   --valid_last


# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name multigpuforecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --test_dir effectivebatchmultinodeforecast_ETTh1_timer_xl_MultivariateDatasetBenchmark_sl4704_it672_ot672_lr0.0001_bt8_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0
done