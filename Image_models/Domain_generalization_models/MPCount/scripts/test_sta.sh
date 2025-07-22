# test stb target from sta source
echo "Testing target: stb from source: sta"
python main.py --task test --config configs/sta_test_stb.yml
# test qnrf target from sta source
echo "Testing target: qnrf from source: sta"
python main.py --task test --config configs/sta_test_qnrf.yml