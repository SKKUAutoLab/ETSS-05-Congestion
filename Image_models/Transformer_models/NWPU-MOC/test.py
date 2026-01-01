import os.path
import warnings
warnings.filterwarnings('ignore')
from importlib import import_module
from models.CC import CrowdCounter
from misc.utils import AverageMeter, AverageCategoryMeter, eval_mc, vis_results
from datasets import createRestore
import torch
import numpy as np
from misc import layer
import datasets
from config import cfg
import argparse

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def multi_class_gauss_map_generate(gt_map):
    gs_layer = getattr(layer, 'Gaussianlayer')
    gs = gs_layer()
    if len(gpu_id) > 1:
        gs = torch.nn.DataParallel(gs, device_ids=gpu_id).cuda()
    else:
        gs = gs.cuda()
    class_gauss_maps = []
    for i in range(len(categorys)):
        class_gt_map = torch.unsqueeze(gt_map[:,i,:,:], 1)
        class_gauss_map = torch.squeeze(gs(class_gt_map), 1)
        class_gauss_maps.append(class_gauss_map)
    gauss_map = torch.stack(class_gauss_maps,1)
    return gauss_map

def save_counting_results(pred_map, gt_map):
    pred_map = pred_map.data.cpu().numpy() # [1, 6, 128, 128]
    gt_map = gt_map.data.cpu().numpy() # [1, 6, 128, 128]
    for c_idx in range(len(categorys)):
        pred_cnt = np.sum(pred_map[:, c_idx, :, :]) / log_para
        gt_count = np.sum(gt_map[:, c_idx, :, :]) / log_para
        print('Category: {}, GT: {:.2f}, Pred: {:.2f}'.format(categorys[c_idx], gt_count, pred_cnt))
        
def test(net, test_loader, args):
    maes = AverageCategoryMeter(len(categorys))
    mses = AverageCategoryMeter(len(categorys))
    cmses = AverageMeter()
    for index, data in enumerate(test_loader, 0):
        for k, v in data.items():
            data[k] = v.cuda()
        rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :args.num_class, :, :] # [1, 3, 1024, 1024], [1, 3, 1024, 1024], [1, 6, 128, 128]
        with torch.set_grad_enabled(False):
            if args.pos_embedding:
                b, c, h, w = rgb.shape
                rh, rw = test_size
                crop_RGBs, crop_Ts, crop_masks = [],[],[]
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_RGBs.append(rgb[:, :, gis:gie, gjs:gje])
                        crop_Ts.append(nir[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros(b, 1, h // label_factor, w // label_factor).cuda()
                        mask[:, :, gis // label_factor:gie // label_factor, gjs // label_factor:gje//label_factor].fill_(1.0)
                        crop_masks.append(mask)
                crop_RGBs, crop_Ts, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_RGBs, crop_Ts, crop_masks))
                crop_data = {'rgb': crop_RGBs, 'nir': crop_Ts,}
                crop_outputs = net(crop_data, args.num_class, mode = 'test')
                crop_preds = crop_outputs['pred_map']
                h, w, rh, rw = h // label_factor, w // label_factor, rh // label_factor, rh // label_factor
                idx = 0
                pred_map = torch.zeros(b, 6, h, w).cuda()
                for i in range(0, h, rh):
                    gis, gie = max(min(h-rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                mask = crop_masks.sum(dim=0)
                pred_map = pred_map / mask # [1, 6, 128, 128]
            else:
                outputs = net(data, args.num_class, mode = 'test')
                pred_map = outputs['pred_map'] # [1, 6, 128, 128]
            abs_errors, square_errors, weights = eval_mc(pred_map, gt_map, log_para)
            wmse = 0.0
            for c_idx in range(len(categorys)):
                maes.update(abs_errors[c_idx], c_idx)
                mses.update(square_errors[c_idx], c_idx)                         
                wmse += square_errors[c_idx] * weights[c_idx]
            cmses.update(wmse)
            save_counting_results(pred_map, gt_map)
            if args.is_vis:
                if not os.path.exists(args.vis_dir):
                    os.makedirs(args.vis_dir)
                mean_std_rgb = ([0.446722, 0.445974, 0.425395], [0.171309, 0.147749, 0.134108])
                # mean_std_ir = ([0.532902, 0.447160, 0.445926], [0.197593, 0.170431, 0.146572])
                vis_results(args.vis_dir, index, createRestore(mean_std_rgb), rgb, pred_map.cpu().detach().numpy()[0, :, :], gt_map.cpu().detach().numpy()[0, :, :], 8)
    overall_mae = maes.avg
    overall_rmse = np.sqrt(mses.avg)
    cls_weight_mse = cmses.avg
    cls_avg_mae = sum(overall_mae) / len(categorys)
    cls_avg_rmse = sum(overall_rmse) / len(categorys)
    print('cls_avg_mae: {:.2f}, cls_avg_mse: {:.2f}, cls_weight_mse: {:.2f}'.format(cls_avg_mae, cls_avg_rmse, cls_weight_mse))
    for c_idx in range(len(categorys)):
        print('Category: {}, MAE: {:.2f}, RMSE: {:.2f}'.format(categorys[c_idx], overall_mae[c_idx], overall_rmse[c_idx]))

def main(args):
    # test loader
    test_loader = datasets.loading_test_data(args.dataset)
    # model
    net = CrowdCounter(args.net_name, gpu_id)
    net.load_state_dict(torch.load(args.ckpt_dir, map_location=lambda storage, loc: storage.cuda(gpu_id[0])), strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    test(net, test_loader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU-MOC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='MOC_RS')
    parser.add_argument('--net_name', type=str, default='MCC')
    parser.add_argument('--pos_embedding', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='saved_nwpu_moc/all_ep_1_cls_avg_mae_11.2_cls_avg_mse_21.4_cls_weight_mse_143.2.pth')
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--is_vis', action='store_true')
    parser.add_argument('--vis_dir', type=str, default='vis_nwpu_moc')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.dataset}')
    cfg_data = datasetting.cfg_data
    gpu_id = [0]
    if len(gpu_id) == 1:
        torch.cuda.set_device(gpu_id[0])
    categorys = cfg_data.CATEGORYS
    log_para = cfg_data.LOG_PARA
    label_factor = cfg_data.LABEL_FACTOR
    test_size = cfg.TRAIN_SIZE
    main(args)