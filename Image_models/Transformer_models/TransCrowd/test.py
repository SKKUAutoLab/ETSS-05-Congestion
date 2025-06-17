from Networks.models import base_patch16_384_token, base_patch16_384_gap
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import setup_seed
import torch
import os
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
import warnings
warnings.filterwarnings('ignore')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pre_data(train_list):
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path)
        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
    return data_keys

def main(args):
    # test list
    if args['type_dataset'] == 'sha':
        test_file = 'npydata/ShanghaiA_test.npy'
    elif args['type_dataset'] == 'shb':
        test_file = 'npydata/ShanghaiB_test.npy'
    elif args['type_dataset'] == 'qnrf':
        test_file = 'npydata/qnrf_test.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    if args['type_model'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['type_model'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)
    else:
        print('This model does not exist')
        raise NotImplementedError
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            print('Load ckpt from: {}'.format(args['pre']))
        else:
            print("No ckpt found at {}".format(args['pre']))
    torch.set_num_threads(args['num_workers'])
    test_data = pre_data(val_list)
    prec1 = validate(test_data, model, args)
    print('Best MAE: {:.4f}'.format(args['best_pred']))

def validate(Pre_data, model, args):
    batch_size = 1
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args['output_dir'], transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False), batch_size=batch_size)
    model.eval()
    mae = 0.0
    mse = 0.0
    for i, (fname, img, gt_count) in enumerate(test_loader):
        img = img.cuda() # [1, 6, 3, 384, 384]
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            out1 = model(img) # [6, 1]
            count = torch.sum(out1).item()
        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        if i % 15 == 0:
            print('{fname}, GT: {gt:.4f}, Pred: {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    nni.report_intermediate_result(mae)
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
    return mae

if __name__ == '__main__':
    setup_seed(args.seed)
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    main(params)