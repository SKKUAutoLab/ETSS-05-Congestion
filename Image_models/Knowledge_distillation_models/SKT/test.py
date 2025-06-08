import os
from models.model_student_vgg import CSRNet as CSRNet_student
from utils import crop_img_patches
import torch
from torch.autograd import Variable
from torchvision import transforms
import argparse
import json
import dataset
import time

def test(test_list, model, args):
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, dataset=args.type_dataset),
                                              shuffle=False, batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        with torch.no_grad():
            output = model(img)
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

def test_ucf(test_list, model, args):
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, dataset=args.type_dataset),
                                              shuffle=False, batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        people = 0
        img_patches = crop_img_patches(img, size=512)
        for patch in img_patches:
            with torch.no_grad():
                sub_output = model(patch)
            people += sub_output.data.sum()
        error = people - target.sum().type(torch.FloatTensor).cuda()
        mae += abs(error)
        mse += error.pow(2)
    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb', 'qnrf'])
    parser.add_argument('--test_json', type=str, default='A_test.json')
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--transform', default=True, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=time.time())
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    model = CSRNet_student(ratio=4, transform=args.transform)
    model = model.cuda()
    # load trained model
    if args.ckpt_dir:
        if os.path.isfile(args.ckpt_dir):
            checkpoint = torch.load(args.ckpt_dir)
            if args.transform is False:
                for k in checkpoint['state_dict'].keys():
                    if k[:9] == 'transform':
                        del checkpoint['state_dict'][k]
            model.load_state_dict(checkpoint['state_dict'])
            print('Load student ckpt from: {}'.format(args.ckpt_dir))
        else:
            print('No student ckpt found')
    if args.type_dataset == 'qnrf':
        test_ucf(test_list, model, args)
    elif args.type_dataset == 'sha' or args.type_dataset == 'shb':
        test(test_list, model, args)
    else:
        print('This dataset does not exist')
        raise NotImplementedError