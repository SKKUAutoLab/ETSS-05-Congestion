import warnings
warnings.filterwarnings("ignore")
import torch
import os
import argparse
from matplotlib import pyplot as plt
from datasets.crowd import Crowd
import numpy as np
from models.SCANet_v7 import SCANet
from utils.evaluation import eval_game, eval_relative

def main(args):
    # test loader
    datasets = Crowd(os.path.join(args.input_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    visual_dir = os.path.join(args.output_dir, 'vis')
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    # model
    model = SCANet()
    model.cuda()
    model_path = os.path.join(args.output_dir, args.ckpt_dir)
    checkpoint = torch.load(model_path, map_location='cuda')['model_state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0
    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
        else:
            inputs = inputs.cuda()
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        img_name = name
        img_name = (str(img_name[0]))[0:4]
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error
            output_vis = outputs[0][0].cpu().numpy()
            target_vis = target[0].cpu().numpy()
            out_count = np.sum(output_vis)
            gt_count = np.sum(target_vis)
            plt.imsave(os.path.join(visual_dir, img_name + '_' + str(gt_count) + '_' + str(out_count) + '.jpg'), output_vis, cmap='magma')
    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    print('GAME0: {:.2f}, GAME1: {:.2f}, GAME2: {:.2f}, GAME3: {:.2f}, MSE: {:.2f}, Relative error: {:.4f}'.format(game[0], game[1], game[2], game[3], mse[0], total_relative_error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='RGBT-CC')
    parser.add_argument('--input_dir', type=str, default='data/processed_RGBT-CC')
    parser.add_argument('--output_dir', type=str, default='saved_rgbt_cc')
    parser.add_argument('--ckpt_dir', type=str, default='best.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)