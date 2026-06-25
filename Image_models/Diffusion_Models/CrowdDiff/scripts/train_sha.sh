DATA_DIR="--data_dir datasets/processed_Shanghaitech/part_A/part_1/train --val_samples_dir datasets/processed_Shanghaitech/part_A/part_1/val"
LOG_DIR="--log_dir results --resume_checkpoint ckpts/64_256_upsampler.pt"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 8 --save_interval 10000 --lr 1e-4"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=0 python scripts/super_res_train.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
