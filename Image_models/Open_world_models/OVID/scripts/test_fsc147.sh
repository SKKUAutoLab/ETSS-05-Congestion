export CUDA_VISIBLE_DEVICES=0
python main.py --mode test --type_dataset FSC --ckpt lightning_logs/ovid_fsc147/version_0/checkpoints/epoch=0-val_mae=51.50.ckpt --backbone b16