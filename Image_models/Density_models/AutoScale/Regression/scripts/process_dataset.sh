python process/density_generate_sh.py --type_dataset ShanghaiTech --input_dir datasets/ShanghaiTech
python process/density_generate_qnrf.py --type_dataset UCF-QNRF --input_dir datasets/UCF-QNRF
python process/density_generate_jhu.py --type_dataset JHU-Crowd++ --input_dir datasets/jhu_crowd_v2.0
python process/density_generate_nwpu.py --type_dataset NWPU-Crowd --input_dir datasets/NWPU_regression
python process/make_npydata.py --sh_dir datasets/ShanghaiTech --qnrf_dir datasets/UCF-QNRF --jhu_dir datasets/jhu_crowd_v2.0 --nwpu_dir datasets/NWPU_regression --output_dir npydata