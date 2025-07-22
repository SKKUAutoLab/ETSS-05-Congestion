# test sta target from stb source
echo "Testing target: sta from source: stb"
python main.py --task test --config configs/stb_test_sta.yml
# test qnrf target from stb source
echo "Testing target: qnrf from source: stb"
python main.py --task test --config configs/stb_test_qnrf.yml