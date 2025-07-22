#export NCCL_P2P_DISABLE=1
#export CUDA_VISIBLE_DEVICES=0,1
#echo "Testing target: sta from source: qnrf"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/qnrf_test_sta.yml --is_ddp
#echo "Testing target: stb from source: qnrf"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/qnrf_test_stb.yml --is_ddp

echo "Testing target: sta from source: qnrf"
python main.py --task test --config configs/qnrf_test_sta.yml
echo "Testing target: stb from source: qnrf"
python main.py --task test --config configs/qnrf_test_stb.yml