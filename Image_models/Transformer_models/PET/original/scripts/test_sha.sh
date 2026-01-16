CUDA_VISIBLE_DEVICES=0
python eval.py --type_dataset sha --input_dir data/ShanghaiTech/part_A --resume saved_sha/best_checkpoint.pth --vis_dir vis_sha
python test_single_image.py --type_dataset sha --input_dir data/ShanghaiTech/part_A/test_data/images/IMG_1.jpg --resume saved_sha/best_checkpoint.pth --vis_dir vis_sha