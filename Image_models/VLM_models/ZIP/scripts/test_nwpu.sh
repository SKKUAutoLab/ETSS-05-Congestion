export CUDA_VISIBLE_DEVICES=0
python test.py --dataset nwpu --weight_path checkpoints/ZIP-P/checkpoints/qnrf/best_mae.pth --input_size 672 --output_filename ebc_p_best_mae --amp --device cuda --local_rank 0
python test.py --dataset nwpu --weight_path checkpoints/ZIP-N/checkpoints/qnrf/best_mae.pth --input_size 672 --output_filename ebc_n_best_mae --amp --device cuda --local_rank 0
python test.py --dataset nwpu --weight_path checkpoints/ZIP-T/checkpoints/qnrf/best_mae.pth --input_size 672 --output_filename ebc_t_best_mae --amp --device cuda --local_rank 0
python test.py --dataset nwpu --weight_path checkpoints/ZIP-S/checkpoints/qnrf/best_mae.pth --input_size 672 --output_filename ebc_s_best_mae --amp --device cuda --local_rank 0
python test.py --dataset nwpu --weight_path checkpoints/ZIP-B/checkpoints/qnrf/best_mae.pth --input_size 672 --output_filename ebc_b_best_mae --amp --device cuda --local_rank 0
