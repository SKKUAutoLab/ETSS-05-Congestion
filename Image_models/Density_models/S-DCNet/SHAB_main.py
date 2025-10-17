import os
from main_process import main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    opt = dict()
    dataset_list = {0: 'SH_partA_Density_map', 1: 'SH_partB_Density_map'}
    model_list = {0: 'model/SHA', 1: 'model/SHB'}
    max_num_list = {0:22, 1:7}
    opt['num_workers'] = 0
    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1
    opt['psize'],opt['pstride'] = 64,64
    set_len = len(dataset_list) # 2
    for ti in range(set_len):
        opt['dataset'] = dataset_list[ti]
        opt['trained_model_path'] = model_list[ti]
        opt['root_dir'] = os.path.join(r'Test_Data',opt['dataset'])
        opt['max_num'] = max_num_list[ti]  
        partition_method = {0:'one_linear',1:'two_linear'}
        opt['partition'] = partition_method[1]
        opt['step'] = 0.5
        print('Testing dataset:', dataset_list[ti])
        main(opt)