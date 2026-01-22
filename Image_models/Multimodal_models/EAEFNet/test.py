import warnings
warnings.filterwarnings("ignore")
import torch
import os
import argparse
from datasets.crowd import Crowd
from models.EAEFNet import fusion_model
from utils.evaluation import eval_game, eval_relative

def main(args):
    # test loader
    datasets = Crowd(os.path.join(args.input_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # model
    model = fusion_model()
    model.cuda()
    model_path = os.path.join(args.output_dir, args.ckpt_dir)
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    print('Load ckpt from:', model_path)
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
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error
    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    print('GAME0: {:.2f}, GAME1: {:.2f}, GAME2: {:.2f}, GAME3: {:.2f}, MSE: {:.2f}, Relative error: {:.4f}'.format(game[0], game[1], game[2], game[3], mse[0], total_relative_error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='RGBT-CC')
    parser.add_argument('--input_dir', type=str, default='data/preprocessed_RGBT-CC')
    parser.add_argument('--output_dir', type=str, default='saved_rgbt_cc')
    parser.add_argument('--ckpt_dir', type=str, default='best_model_0.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)