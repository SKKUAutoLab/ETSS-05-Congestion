export CUDA_VISIBLE_DEVICES=0 
python run.py --mode test --exp_name exp --batch_size 32 --dataset_type FSC --ckpt ckpts/clipcount_pretrained.ckpt
