export CUDA_VISIBLE_DEVICES=0
python main.py --mode train --type_dataset ShanghaiTech --batch_size 8 --epochs 200 --backbone b16 --output_dir saved_shanghaitech