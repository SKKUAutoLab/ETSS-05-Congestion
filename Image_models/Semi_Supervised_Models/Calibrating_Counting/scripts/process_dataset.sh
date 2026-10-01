ln -sfn datasets/ShanghaiTech/part_A part_A
mkdir -p part_A/uncertain_data
mkdir -p part_A/uncertain_label
mkdir -p part_A/uncertain_test
mkdir -p part_A/uncertain_data_5
mkdir -p part_A/uncertain_data_10
mkdir -p part_A/uncertain_data_40
python patch_gen_A.py
python patch_move.py
