python process/distance_generate_SH.py --type_dataset ShanghaiTech --input_dir datasets/ShanghaiTech
python process/distance_generate_QNRF.py --type_dataset UCF-QNRF --input_dir datasets/UCF-QNRF
python process/distance_generate_jhu.py --type_dataset JHU-Crowd++ --input_dir datasets/jhu_crowd_v2.0
python process/distance_generate_nwpu.py --type_dataset NWPU-Crowd --input_dir datasets/NWPU_localization
python process/make_npydata.py --sh_dir datasets/ShanghaiTech --qnrf_dir datasets/UCF-QNRF --jhu_dir datasets/jhu_crowd_v2.0 --nwpu_dir datasets/NWPU_localization --output_dir npydata