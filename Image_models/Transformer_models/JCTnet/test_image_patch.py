import argparse
import torch
import os
import numpy as np
import datasets.crowd_UCF_CC as crowd
from Networks.Swin import SwinIR
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')
parser.add_argument('--crop-size', type=int, default=256,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='/home/xuzhiwen/home/xuzhiwen/dataset_wfs/TransDensityNet/'\
                    'ckpts/VGG1-8-PartA+SwinIR/12-29-input-256-lr-1e-05-Count/best_model_mae 62.20.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='/home/hdd/dataset_xzw/TRANCOS_Nor',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='trancso',
                    help='dataset name: qnrf, nwpu, sha, shb, trancso')
parser.add_argument('--pred-density-map-path', type=str, default='',
                    help='save predicted density maps when pred-density-map-path is not empty.')

def test(args, isSave = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'trancso':
        dataset = crowd.Crowd_trancos(os.path.join(data_path, 'test'), crop_size, 8, 'test')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    model = SwinIR.Net()
    model.to(device)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()
    image_errs = []
    result = []
    for inputs, count, name in dataloader:
        with torch.no_grad():
            # nputs = cal_new_tensor(inputs, min_size=args.crop_size)
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred = model(crop_imgs[gs:gt])

                _, _, h1, w1 = crop_pred.size()
                crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask

            img_err = count[0].item() - torch.sum(outputs).item()
            print(name, img_err, count[0].item(), torch.sum(outputs).item())
            image_errs.append(img_err)
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))

    if isSave:
        with open("SwinIR_trancso_test.txt","w") as f:
            for i in range(len(result)):
                f.write(str(result[i]).replace('[','').replace(']','').replace(',', ' ')+"\n")
            f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave= True)


