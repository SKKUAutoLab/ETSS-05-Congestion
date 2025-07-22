#export NCCL_P2P_DISABLE=1
#export CUDA_VISIBLE_DEVICES=0,1
#echo "Testing target: stb from source: sta"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/sta_test_stb.yml --is_ddp
#echo "Testing target: qnrf from source: sta"
#torchrun --nnodes=1 --nproc_per_node=2 main.py --task test --config configs/sta_test_qnrf.yml --is_ddp

echo "Testing target: stb from source: sta"
python main.py --task test --config configs/sta_test_stb.yml
echo "Testing target: qnrf from source: sta"
python main.py --task test --config configs/sta_test_qnrf.yml