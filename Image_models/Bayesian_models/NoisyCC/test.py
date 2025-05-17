import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/qnrf', type=str)
    parser.add_argument('--ckpt_dir', default='saved_qnrf/best_val.pth', type=str)
    args = parser.parse_args()

    print('Testing dataset:', args.input_dir.split('/')[-1])
    # test loader
    datasets = Crowd(os.path.join(args.input_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=1, pin_memory=False)
    # load trained model
    model = vgg19()
    model.cuda()
    model.load_state_dict(torch.load(args.ckpt_dir, map_location='cuda'))
    # test
    epoch_minus = []
    for inputs, count, name in dataloader:
        inputs = inputs.cuda() # [1, 3, 1875, 1500]
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs) # [1, 1, 234, 312]
            temp_minu = len(count[0]) - torch.sum(outputs).item()
            epoch_minus.append(temp_minu)
    epoch_minus = np.array(epoch_minus) # [334]
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))