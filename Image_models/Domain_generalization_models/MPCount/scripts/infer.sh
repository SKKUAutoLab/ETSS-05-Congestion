# infer target qnrf from source sta
python inference.py --img_path data/qnrf/test/img_0001.jpg --model_path logs/sta/best.pth --save_path output_src_sta.txt --vis_dir vis
# infer target qnrf from source stb
python inference.py --img_path data/qnrf/test/img_0002.jpg --model_path logs/stb/best.pth --save_path output_src_stb.txt --vis_dir vis