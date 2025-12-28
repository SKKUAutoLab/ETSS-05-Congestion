export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 --use_env main.py --type_dataset HT21 --input_dir data/HT21 --output_dir saved_ht21