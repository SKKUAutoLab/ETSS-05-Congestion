mkdir -p history/test/eval_61
cp -f history/test/best_model_61.pth history/test/eval_61/
python test.py --data-dir datasets/sha/val --save-dir history/test/eval_61 --device 0 --visual-img --vis_dir saved_vis_sha
