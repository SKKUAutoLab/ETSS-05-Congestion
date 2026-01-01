from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.DATASET = 'MOC_RS'
__C.MM = False
__C.NET = 'MCC'
__C.PRE_BACKBONE_WEIGHT = ''
__C.POS_EMBEDDING = False
__C.PRE_GCC = False
__C.PRE_GCC_MODEL = ''
__C.RESUME = False
__C.RESUME_PATH = ''
__C.BACKBONE_FREEZE = False
__C.GPU_ID = [0] # [0,1]
__C.TRAIN_SIZE = (512,512)
__C.TRAIN_BATCH_SIZE = 4
__C.BASE_LR = 5*1e-5
__C.CONV_LR = 1e-6
__C.WEIGHT_DECAY = 1e-4
__C.LOSS_FUNCTION = 'Mse_loss' # Mse_loss, Cos_loss, Mix_loss, Mask_loss
__C.MODEL_ARCH = dict(backbone=dict(embed_dim=128, depths=[2, 2, 18], num_heads=[4, 8, 16], out_indices=(0, 1, 2), window_size=7, ape=False, drop_path_rate=0.3, patch_norm=True,
                                    use_checkpoint=False), decode_head=dict(in_channels=[128, 256, 512]))
__C.PRE_WEIGHTS = 'pretrained_models/swin_base_patch4_window7_224.pth'
__C.MAX_EPOCH = 200
__C.alpha = 1
__C.PRINT_FREQ = 20
__C.EXP_PATH = 'saved_nwpu_moc'
__C.VAL_STAGE = [0,50,100,200]
__C.VAL_FREQ = [10,4,4,4]
__C.CP_FREQ = 10