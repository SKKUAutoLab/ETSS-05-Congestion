export CUDA_VISIBLE_DEVICES=0 
python run.py --mode test --exp_name exp --batch_size 32 --dataset_type FSC --ckpt ckpt/epoch=149-avg_fine_accuracy_pred=0.71.ckpt
