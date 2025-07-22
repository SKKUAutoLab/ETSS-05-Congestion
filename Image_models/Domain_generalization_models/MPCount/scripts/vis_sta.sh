# vis stb target from sta source
echo "Visualizing target: stb from source: sta"
python main.py --task vis --config configs/sta_test_stb.yml
# vis qnrf target from sta source
echo "Visualizing target: qnrf from source: sta"
python main.py --task vis --config configs/sta_test_qnrf.yml