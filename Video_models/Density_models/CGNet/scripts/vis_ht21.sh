ulimit -n 1000000
python inference.py --type_dataset HT21 --pair_config configs/crowd_ht21.json --ckpt_dir saved_ht21/checkpoints/best.pth --vis_dir saved_ht21/vis --is_vis --interval 3