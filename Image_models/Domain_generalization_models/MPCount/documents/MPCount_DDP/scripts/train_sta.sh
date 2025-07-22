export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nnodes=1 --nproc_per_node=2 main.py --task train --config configs/sta_train.yml --is_ddp