import torch
from torch.utils.data import  DataLoader
import os
import numpy as np
from load_data_V2 import myDataset, ToTensor
from Network.SDCNet import SDCNet_VGG16_classify
from Val import test_phase

def main(opt):
    root_dir = opt['root_dir']
    num_workers = opt['num_workers']
    img_subsubdir = 'images'
    tar_subsubdir = 'gtdens'
    dataset_transform = ToTensor()
    if opt['partition'] =='one_linear':
        label_indice = np.arange(opt['step'],opt['max_num']+opt['step']/2,opt['step'])
        add = np.array([1e-6])
        label_indice = np.concatenate( (add,label_indice) )
    elif opt['partition'] =='two_linear':
        label_indice = np.arange(opt['step'], opt['max_num'] + opt['step'] / 2, opt['step']) # [44]
        add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) # [10]
        label_indice = np.concatenate((add,label_indice)) # [54]
    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size+1
    img_dir = os.path.join(root_dir,'test',img_subsubdir)
    tar_dir = os.path.join(root_dir,'test',tar_subsubdir)
    rgb_dir = os.path.join(root_dir,'rgbstate.mat')
    testset = myDataset(img_dir, tar_dir, rgb_dir, transform=dataset_transform, if_test=True, IF_loadmem=opt['IF_savemem_test'])
    testloader = DataLoader(testset, batch_size=opt['test_batch_size'], shuffle=False, num_workers=num_workers)
    label_indice = torch.Tensor(label_indice) # [54]
    class_num = len(label_indice) + 1
    div_times = 2
    # model
    net = SDCNet_VGG16_classify(class_num,label_indice,psize=opt['psize'], pstride=opt['pstride'], div_times=div_times, load_weights=True).cuda()
    mod_path = 'best_epoch.pth'
    mod_path = os.path.join(opt['trained_model_path'],mod_path)
    if os.path.exists(mod_path):
        all_state_dict = torch.load(mod_path)
        net.load_state_dict(all_state_dict['net_state_dict'])
        log_save_path = os.path.join(opt['trained_model_path'],'log-trained-model.txt')
        mae, rmse, me = test_phase(opt, net, testloader, log_save_path=log_save_path)
        log_str = '%10s\t %8s\t &%8s\t &%8s\t\\\\' % (' ','mae','rmse','me')+'\n'
        log_str += '%-10s\t %8.3f\t %8.3f\t %8.3f\t' % ( 'test',mae,rmse,me ) + '\n'
        print(log_str)