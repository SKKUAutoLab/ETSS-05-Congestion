# train with single GPU
# python train.py --type_dataset HT21 --config configs/crowd_ht21.json
# train with multiple GPU
torchrun --master_port 29505 --nproc_per_node=2 train.py --config configs/crowd_ht21.json --type_dataset HT21