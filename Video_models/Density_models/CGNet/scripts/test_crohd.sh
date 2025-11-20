ulimit -n 100000
python inference.py --type_dataset HT21 --pair_config configs/crowd_ht21.json --ckpt_dir saved_ht21/checkpoints/best.pth --interval 75