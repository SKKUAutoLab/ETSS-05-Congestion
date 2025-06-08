python preprocess/ShanghaiTech_GT_generation.py --input_dir datasets/ShanghaiTech
python preprocess/UCF_GT_generation.py --input_dir datasets/UCF-QNRF
python preprocess/make_json.py --type_dataset sha --input_dir datasets/ShanghaiTech/part_A_final --train_json A_train.json --val_json A_val.json --test_json A_test.json
python preprocess/make_json.py --type_dataset shb --input_dir datasets/ShanghaiTech/part_B_final --train_json B_train.json --val_json B_val.json --test_json B_test.json
python preprocess/make_json.py --type_dataset qnrf --input_dir datasets/UCF-QNRF --train_json qnrf_train.json --val_json qnrf_val.json --test_json qnrf_test.json