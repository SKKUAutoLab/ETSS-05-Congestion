python utils/preprocess_data.py --type_dataset sta --input_dir data/ShanghaiTech/part_A --output_dir data/sta
python utils/preprocess_data.py --type_dataset stb --input_dir data/ShanghaiTech/part_B --output_dir data/stb
python utils/preprocess_data.py --type_dataset qnrf --input_dir data/UCF-QNRF --output_dir data/qnrf
python utils/dmap_gen.py --input_dir data/sta
python utils/dmap_gen.py --input_dir data/stb
python utils/dmap_gen.py --input_dir data/qnrf