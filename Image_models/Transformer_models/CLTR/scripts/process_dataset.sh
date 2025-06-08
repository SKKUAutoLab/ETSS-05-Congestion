python data/prepare_jhu.py --input_dir data/jhu_crowd_v2.0
python data/prepare_nwpu.py --input_dir data/NWPU_CLTR
python make_npydata.py --jhu_path data/jhu_crowd_v2.0 --nwpu_path data/NWPU_CLTR --output_dir npydata