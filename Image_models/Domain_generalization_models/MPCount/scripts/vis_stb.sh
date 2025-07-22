# vis sta target from stb source
echo "Visualizing target: sta from source: stb"
python main.py --task vis --config configs/stb_test_sta.yml
# vis qnrf target from stb source
echo "Visualizing target: qnrf from source: stb"
python main.py --task vis --config configs/stb_test_qnrf.yml