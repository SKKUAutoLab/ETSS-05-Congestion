echo "Testing target: sta from source: qnrf"
python main.py --task test --config configs/qnrf_test_sta.yml
echo "Testing target: stb from source: qnrf"
python main.py --task test --config configs/qnrf_test_stb.yml