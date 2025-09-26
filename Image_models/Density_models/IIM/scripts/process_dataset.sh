python process/prepare_SHHA.py --type_dataset SHHA --input_dir data/ShanghaiTech/part_A --output_dir data/SHHA
python process/prepare_SHHB.py --type_dataset SHHB --input_dir data/ShanghaiTech/part_B --output_dir data/SHHB
python process/prepare_QNRF.py --type_dataset QNRF --input_dir data/UCF-QNRF --output_dir data/QNRF
python process/prepare_NWPU.py --type_dataset NWPU --input_dir data/NWPU-Crowd
python process/prepare_JHU.py --type_dataset JHU --input_dir data/jhu_crowd_v2.0 --output_dir data/JHU