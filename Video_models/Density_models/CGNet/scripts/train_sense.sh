# train with single GPU
# python train.py --type_dataset SENSE --config configs/crowd_sense.json
# train with multiple GPU
torchrun --master_port 29505 --nproc_per_node=2 train.py --config configs/crowd_sense.json --type_dataset SENSE