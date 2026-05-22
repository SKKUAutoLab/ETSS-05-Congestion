export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
python demo_suwon.py --video_path example.mp4 --num_queries 700 --pre saved_suwon/model_best.pth --threshold 0.1 --gpu_id 0
