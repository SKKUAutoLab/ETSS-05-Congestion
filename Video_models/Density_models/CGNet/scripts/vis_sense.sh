ulimit -n 1000000
python inference.py --type_dataset SENSE --pair_config configs/crowd_sense.json --ckpt_dir saved_sense/checkpoints/best.pth --vis_dir saved_sense/vis --is_vis --interval 3