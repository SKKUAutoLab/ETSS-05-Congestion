export CUDA_VISIBLE_DEVICES=0
python test.py --type_dataset SENSE --input_dir data/Sense/test --ann_dir data/Sense/label_list_all --interval 1 --vis_dir saved_sense/vis --is_vis --output_dir saved_sense
python convert_gif.py --type_dataset SENSE --input_dir saved_sense/vis --output_dir saved_sense_gif