export CUDA_VISIBLE_DEVICES=0
python test.py --dataset shb --weight_path checkpoints/ZIP-P/checkpoints/shb/best_mae.pth --output_filename ebc_p_best_mae --amp --local_rank 0
python test.py --dataset shb --weight_path checkpoints/ZIP-N/checkpoints/shb/best_mae.pth --output_filename ebc_n_best_mae --amp --local_rank 0
python test.py --dataset shb --weight_path checkpoints/ZIP-T/checkpoints/shb/best_mae.pth --output_filename ebc_t_best_mae --amp --local_rank 0
python test.py --dataset shb --weight_path checkpoints/ZIP-S/checkpoints/shb/best_mae.pth --output_filename ebc_s_best_mae --amp --local_rank 0
python test.py --dataset shb --weight_path checkpoints/ZIP-B/checkpoints/shb/best_mae.pth --output_filename ebc_b_best_mae --amp --local_rank 0
