#export NCCL_P2P_DISABLE=1
#export CUDA_VISIBLE_DEVICES=0,1
#echo "Testing target: sta from source: stb"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/stb_test_sta.yml --is_ddp
#echo "Testing target: qnrf from source: stb"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/stb_test_qnrf.yml --is_ddp

echo "Testing target: sta from source: stb"
python main.py --task test --config configs/stb_test_sta.yml
echo "Testing target: qnrf from source: stb"
python main.py --task test --config configs/stb_test_qnrf.yml