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
token_num=32
token_len=96
# seq_len=$[$token_num*$token_len]
# 修复数组计算语法
seq_len=$((token_num * token_len))

# 分布式训练参数
DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --node_rank $node_rank --nnodes $nnodes --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# 使用torch.distributed.launch启动训练
python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/era5_pretrain/ \
  --data_path pretrain.npy \
  --model_id era5_pretrain \
  --model $model_name \
  --data Era5_Pretrain  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 40960 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --devices 0,1 \
  --ddp 