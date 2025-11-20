cd CrowdLocater
python create_json.py --type_dataset SENSE --input_dir ../datasets/Sense --output_dir datasets/SENSE
python inference.py --type_dataset SENSE --config fidtm_sense.json --mode test --ckpt_dir ckpts/weight.pth --draw_dir ../locater/vis --output_dir ../locater/results # --vis
python create_json.py --type_dataset HT21 --input_dir ../datasets/HT21 --output_dir datasets/HT21
python inference.py --type_dataset HT21 --config fidtm_ht21.json --mode test --ckpt_dir ckpts/weight.pth --draw_dir ../locater/vis --output_dir ../locater/results --vis
cd ..