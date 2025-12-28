export CUDA_VISIBLE_DEVICES=0
python test_ht21.py --type_dataset HT21 --input_dir data/HT21/test --interval 3 --vis_dir saved_ht21/vis --is_vis --output_dir saved_ht21
python convert_gif.py --type_dataset HT21 --input_dir saved_ht21/vis --output_dir saved_ht21_gif