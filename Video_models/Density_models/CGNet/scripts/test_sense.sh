ulimit -n 1000000
python inference.py --type_dataset SENSE --pair_config configs/crowd_sense.json --ckpt_dir saved_sense/checkpoints/best.pth
python eval.py --type_dataset SENSE --input_dir saved_sense/video_results_test.json --ann_dir datasets/Sense/label_list_all