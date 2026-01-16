import warnings
warnings.filterwarnings("ignore")
from scipy import ndimage
import argparse
import numpy as np
import os
import random
import math
from models.contrastive_loss import ContrastiveLoss
import torch
import torch.nn.functional as F
from typing import Dict, Any
import util.misc as misc
from datasets.FSC147 import FSC147
from datasets.ShanghaiTech import ShanghaiTech
from models import ovid
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from util.misc import collate_fn
from models.matcher_loss import criterion
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def setup_seed(seed):
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)
        self.criterion = criterion()
        self.save_hyperparameters(args)
        self.model = ovid.OVID(fim_depth=self.args.decoder_depth, fim_num_heads=self.args.decoder_head, use_coop=self.args.use_coop, use_vpt=self.args.use_vpt,
                               coop_width=self.args.coop_width, vpt_width=self.args.vpt_width, vpt_depth= self.args.vpt_depth, backbone = self.args.backbone,
                               use_fim = self.args.use_fim, use_mixed_fim = self.args.use_mixed_fim, unfreeze_vit = self.args.unfreeze_vit)
        self.loss = F.mse_loss
        self.contrastive_loss = ContrastiveLoss(0.07, self.args.normalize_contrast)
        self.neg_prompt_embed = None

    def training_step(self, batch, batch_idx):
        samples, gt_density, dots, boxes, m_flag, prompt_gt, prompt_add = batch # [8, 3, 384, 384], [8, 384, 384], [8], [8, 3, 3, 64, 64], [8], [8], [8, 5]
        # output: [8, 9216, 2], [8, 9216, 2], extra_out: [8, 1, 512], [8, 1, 512], [8, 196, 512], [8, 196, 512], [8, 512, 28, 28]
        output, extra_out = self.model(samples, prompt_gt, return_extra=True, coop_require_grad=True)
        loss_dict, src_points = self.criterion.forward_and_pair(output, dots)
        weight_dict= self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        pred_density = np.zeros(gt_density.shape) # [8, 384, 384]
        for i in range(len(src_points)):
            for j in range(len(src_points[i])):
                pred_density[i][min(gt_density.shape[2] - 1, int(src_points[i][j][1]))][min(gt_density.shape[1] - 1, int(src_points[i][j][0]))] = 1
        reresized_pred_density = np.zeros(gt_density.shape) # [8, 384, 384]
        for i in range(pred_density.shape[0]):
            reresized_pred_density[i] = ndimage.gaussian_filter(pred_density[i], sigma=(1, 1), order=0) * 60
        reresized_pred_density = torch.tensor(reresized_pred_density,device=self.device) # [8, 384, 384]
        mask = np.random.binomial(n=1, p=0.8, size=[384, 384]) # [384, 384]
        masks = np.tile(mask,(reresized_pred_density.shape[0], 1)) # [3072, 384]
        masks = masks.reshape(reresized_pred_density.shape[0], 384, 384) # [8, 384, 384]
        masks = torch.from_numpy(masks).to(self.device)
        loss = self.loss(reresized_pred_density, gt_density)
        loss = (loss * masks / (384 * 384)).sum() / reresized_pred_density.shape[0]
        if self.args.use_contrast and self.current_epoch <= self.args.contrast_pre_epoch:
            text_embedding = extra_out['text_embedding'] # [8, 1, 512]
            if self.args.contrast_pos == "pre":
                patch_embedding = extra_out['patch_embedding_contrast'] # [8, 196, 512]
            elif self.args.contrast_pos == "post":
                patch_embedding = extra_out['pixel_text_matching_map']
            contrast_loss = self.contrastive_loss(patch_embedding, text_embedding, gt_density.detach().clone())
            loss = args.w_contrast * contrast_loss
            self.log('train_loss_contrast', contrast_loss)
        loss = 0.75 * losses + 0.25 * loss
        self.log('train_loss', loss)
        batch_mae = 0
        gt_cnts = 0
        pred_cnts = 0
        batch_rmse = 0
        batch_size_temp = output['pred_points'].shape[0]
        for i in range(output['pred_points'].shape[0]):
            outputs_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1][i] # [9216]
            gt_cnt = dots[i]['point'].shape[0]
            threshold = 0.5
            predict_cnt = int((outputs_scores > threshold).sum())
            cnt_err = abs(predict_cnt - gt_cnt)
            batch_mae += cnt_err
            batch_rmse += cnt_err**2
            pred_cnts += predict_cnt
            gt_cnts += gt_cnt
        batch_mae /= batch_size_temp
        batch_rmse /= batch_size_temp
        batch_rmse = math.sqrt(batch_rmse)
        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)
        return loss
    
    def validation_step(self, batch, batch_idx):
        samples, gt_density, dots, _, _, prompt, _ = batch # [1, 3, 384, 384], [1, 384, 384], [1], ['bottle caps']
        if not self.args.use_contrast:
            prompt = [f"a photo of {p}" for p in prompt]
        output = self.model(samples, prompt) # [1, 9216, 2], [1, 9216, 2]
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        outputs_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1][0] # [9216]
        gt_cnt = dots['point'].shape[1]
        threshold = 0.5
        predict_cnt = int((outputs_scores > threshold).sum())
        cnt_err=abs(predict_cnt - gt_cnt)
        batch_mae.append(cnt_err)
        batch_rmse.append(cnt_err ** 2)
        pred_cnts.append(predict_cnt)
        gt_cnts.append(gt_cnt)
        return {"mae": batch_mae, "rmse": batch_rmse, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}

    def validation_epoch_end(self, outputs):
        all_mae = []
        all_rmse = []
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)
        idx = random.randint(0, len(outputs)-1)
        prompt = outputs[idx]["prompt"]
        pred_cnts = outputs[idx]["pred_cnts"]
        gt_cnts = outputs[idx]["gt_cnts"]
        pred_gt = "pred: {:.2f} gt: {:.2f}".format(pred_cnts[0], gt_cnts[0])
        self.logger.experiment.add_text("prompt", prompt, self.current_epoch)
        self.logger.experiment.add_text("count", pred_gt, self.current_epoch)
        print('Prompt: {}, Pred: {:.2f}, GT: {:.2f}'.format(prompt, pred_cnts[0], gt_cnts[0]))
    
    def test_step(self, batch, batch_idx):
        if self.args.type_dataset == 'FSC':
            image, gt_density, dots, boxes, m_flag, prompt = batch # [1, 3, 384, 656], [1, 384, 656], [1, 28, 2], [1, 3, 3, 64, 64], [3], [1]
        elif self.args.type_dataset == "ShanghaiTech":
            image, gt_cnt = batch # [1, 3, 384, 512], 15
            gt_cnt = gt_cnt.item()
            prompt = ["people" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3]) # [1, 384, 512]
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        assert image.shape[0] == 1, "only support inference one image at a time"
        patches, _ = misc.sliding_window(image, stride=384) # [2, 3, 384, 384]
        patches = torch.from_numpy(patches).float().to(self.device) # [2, 3, 384, 384]
        prompt = np.repeat(prompt, patches.shape[0], axis=0) # [2]
        output = self.model(patches, prompt) # [2, 9216, 2], [2, 9216, 2]
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        outputs_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1] # [2, 9216]
        if self.args.type_dataset == 'FSC':
            gt_cnt = dots['point'].shape[1]
        threshold = 0.5
        predict_cnt = int((outputs_scores > threshold).sum())
        cnt_err = abs(predict_cnt - gt_cnt)
        batch_mae.append(cnt_err)
        batch_rmse.append(cnt_err ** 2)
        pred_cnts.append(predict_cnt)
        gt_cnts.append(gt_cnt)
        return {"mae": batch_mae, "rmse": batch_rmse, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}

    def test_epoch_end(self, outputs):
        all_mae = []
        all_rmse = []
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        test_mae = np.mean(all_mae)
        test_rmse = np.sqrt(np.mean(all_rmse))
        self.log('test_mae', test_mae)
        self.log('test_rmse', test_rmse)
        print('Test MAE: {:.2f}, Test MSE: {:.2f}'.format(test_mae, test_rmse))

    def forward(self, img, prompt):
        return self.model(img, prompt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.95), weight_decay=self.args.weight_decay)
        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.args.unfreeze_vit :
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("model.clip") or k.startswith("model.img_encoder.clip") or k.startswith("model.text_encoder.clip") or k.startswith("model.img_encoder.vit"):
                    del checkpoint["state_dict"][k]

    def overwrite_args(self, args):
        self.args = args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FSC', choices=['FSC', 'ShanghaiTech'])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="saved_fsc147")
    parser.add_argument('--seed', default=42, type=int)
    # training config
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--val_freq', default=5, type=int)
    # model config
    parser.add_argument('--backbone', default="b16", choices=["b16", "b32", "l14"])
    parser.add_argument('--decoder_depth', default=4, type=int) # num of FIM layers
    parser.add_argument('--decoder_head', default=8, type=int) # num of att heads for FIM
    parser.add_argument('--use_mixed_fim', default=True, type=misc.str2bool) # use hierarchical patch-text interaction
    parser.add_argument('--unfreeze_vit', default=False, type=misc.str2bool) # unfreeze or finetune clip
    parser.add_argument('--use_fim', default=False, type=misc.str2bool) # whether to use naive interaction
    # loss config
    parser.add_argument('--use_coop', default=True, type=misc.str2bool) # context learning for text prompts
    parser.add_argument('--coop_width', default=2, type=int) # num tokens to be learned
    parser.add_argument('--coop_require_grad', default=False, type=misc.str2bool)
    parser.add_argument('--use_vpt', default=True, type=misc.str2bool)
    parser.add_argument('--vpt_width', default=20, type=int)
    parser.add_argument('--vpt_depth', default=10, type=int)
    parser.add_argument("--use_contrast", default=True, type=misc.str2bool)
    parser.add_argument("--w_contrast", default=1.0, type=float)
    parser.add_argument('--normalize_contrast', default=False, type=misc.str2bool)
    parser.add_argument('--contrast_pos', default="pre", choices=["pre", "post"], type=str)
    parser.add_argument('--contrast_pre_epoch', default=20, type=int)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    # train and test loader
    dataset_train = FSC147(split="train")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                   drop_last=False, collate_fn=collate_fn)
    dataset_val = FSC147(split="val")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader =  torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=4, mode='min', filename='{epoch}-{val_mae:.2f}')
    # model
    model = Model(args)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.output_dir)
    trainer = Trainer(accelerator="gpu",  callbacks=[save_callback], accumulate_grad_batches = args.accum_iter, precision=16, max_epochs=args.epochs + args.contrast_pre_epoch,
                      logger=logger, check_val_every_n_epoch=args.val_freq)
    if args.mode == "train":
        if args.ckpt is not None:
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
        trainer.fit(model, train_dataloader, val_dataloader)
    elif args.mode == "test":
        if args.type_dataset == "FSC":
            dataset_val = FSC147(split="val", resize_val=False)
            dataset_test = FSC147(split="test")
        elif args.type_dataset == "ShanghaiTech":
            dataset_val = dataset_test = ShanghaiTech(split="test", part="B")
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        val_dataloader = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt, strict=False)
        model.overwrite_args(args)
        model.eval()
        if args.type_dataset == "FSC":
            print('Validation results')
            trainer.test(model, val_dataloader)
        print('Testing results')
        trainer.test(model, test_dataloader)