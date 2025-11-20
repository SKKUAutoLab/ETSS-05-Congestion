import numpy as np
import json
import os
import argparse

def main(args):
    with open(args.input_dir, "r") as f:
        video_results = json.load(f)
    gt_video_num_list = []
    gt_video_len_list = []
    pred_video_num_list = []
    for video_name in video_results:
        anno_path = os.path.join(args.ann_dir, video_name + ".txt")
        with open(anno_path, "r") as f:
            lines = f.readlines()
            all_ids = set()
            for line in lines:
                line = line.strip().split(" ")
                data = [float(x) for x in line[3:] if x != ""]
                if len(data) > 0:
                    data = np.array(data)
                    data = np.reshape(data, (-1, 7))
                    ids = data[:, 6].reshape(-1, 1)
                    for id in ids:
                        all_ids.add(int(id[0]))
        info = video_results[video_name]
        gt_video_num = len(all_ids)
        pred_video_num = info["video_num"]
        pred_video_num_list.append(pred_video_num)
        gt_video_num_list.append(gt_video_num)
        gt_video_len_list.append(info["frame_num"])
    MAE = np.mean(np.abs(np.array(gt_video_num_list)-np.array(pred_video_num_list)))
    MSE = np.mean(np.square(np.array(gt_video_num_list)-np.array(pred_video_num_list)))
    WRAE = np.sum(np.abs(np.array(gt_video_num_list)-np.array(pred_video_num_list)) * np.array(gt_video_len_list) / np.array(gt_video_num_list) / np.sum(gt_video_len_list))
    RMSE = np.sqrt(MSE)
    print(f"MAE: {MAE:.2f}, MSE: {MSE:.2f}, WRAE: {WRAE * 100:.2f}%, RMSE: {RMSE:.2f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--input_dir', type=str, default='saved_sense/video_results_test.json')
    parser.add_argument('--ann_dir', type=str, default='datasets/Sense/label_list_all')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)