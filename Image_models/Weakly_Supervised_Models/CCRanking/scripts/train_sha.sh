python train.py --dataset_name ShanghaiTechA --eval_mode Rank --end_step 1000 --eval_step 1 --experiment-ID 1 --gpu_id 1
python train_baseline.py --dataset_name ShanghaiTechA_Combine --compare_loss_mode --eval_mode Rank --end_step 1000 --eval_step 1 --experiment-ID eval_1 --gpu_id 1 --lambda_reg 0.2 --lr 0.000005 --save_all_weights
