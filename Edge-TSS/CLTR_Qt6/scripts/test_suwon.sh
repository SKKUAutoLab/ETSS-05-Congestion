export NCCL_P2P_DISABLE=1
python test.py --type_dataset suwon --pre saved_suwon/checkpoint.pth --gpu_id 0 --num_queries 700 --output_dir saved_suwon --fp16