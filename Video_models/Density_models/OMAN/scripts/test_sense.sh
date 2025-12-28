export CUDA_VISIBLE_DEVICES=0
python test.py --type_dataset SENSE --input_dir data/Sense/test --ann_dir data/Sense/label_list_all --output_dir saved_sense
python eval_metrics.py --type_dataset SENSE --output_dir saved_sense --ann_dir data/Sense/label_list_all --label_dir data/Sense/scene_label.txt