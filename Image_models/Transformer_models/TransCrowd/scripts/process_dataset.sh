python data/predataset_sh.py
python data/predataset_qnrf.py
python make_npydata.py --type_dataset sha --input_dir datasets/ShanghaiTech
python make_npydata.py --type_dataset shb --input_dir datasets/ShanghaiTech
python make_npydata.py --type_dataset qnrf --input_dir datasets/UCF-QNRF_ECCV18