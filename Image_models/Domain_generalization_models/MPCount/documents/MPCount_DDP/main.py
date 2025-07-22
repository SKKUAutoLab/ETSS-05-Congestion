import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import yaml
import argparse
from trainers.dgtrainer import DGTrainer
from models.models import DGModel_final
from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset
from utils.misc import seed_worker, get_seeded_generator, seed_everything
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_model(name, params):
    if name == 'final':
        return DGModel_final(**params)
    else:
        print('This model does not exist')
        raise NotImplementedError

def get_loss():
    return nn.MSELoss()

def get_dataset(name, params, method):
    if name == 'den':
        dataset = DensityMapDataset(method=method, **params)
        collate = DensityMapDataset.collate
    elif name == 'den_cls':
        dataset = DenClsDataset(method=method, **params)
        collate = DenClsDataset.collate
    elif name == 'jhu_domain':
        dataset = JHUDomainDataset(method=method, **params)
        collate = JHUDomainDataset.collate
    elif name == 'jhu_domain_cls':
        dataset = JHUDomainClsDataset(method=method, **params)
        collate = JHUDomainClsDataset.collate
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    return dataset, collate

def get_optimizer(name, params, model):
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **params)
    else:
        print('This optimizer does not exist')
        raise NotImplementedError

def get_scheduler(name, params, optimizer):
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
    else:
        print('This scheduler does not exist')
        raise NotImplementedError

def load_config(config_path, task, is_ddp):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # setup DDP training
    if is_ddp:
        dist.init_process_group("nccl")
        assert cfg['train_loader']['batch_size'] % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = 'cuda:0'
        torch.cuda.set_device(device)
    # setup other configs
    init_params = {}
    task_params = {}
    if is_ddp:
        init_params['seed'] = cfg['seed'] * dist.get_world_size() + rank
    else:
        init_params['seed'] = cfg['seed']
    init_params['version'] = cfg['version']
    init_params['log_para'] = cfg['log_para']
    init_params['patch_size'] = cfg['patch_size']
    init_params['mode'] = cfg['mode']
    if is_ddp:
        seed_everything(init_params['seed'])
    else:
        seed_everything(cfg['seed'])
    task_params['model'] = get_model(cfg['model']['name'], cfg['model']['params'])
    if is_ddp:
        task_params['model'] = DDP(task_params['model'].to(device), device_ids=[device])
        init_params['rank'] = rank
        init_params['device'] = device
    else:
        task_params['model'] = task_params['model'].to(device)
        init_params['rank'] = rank
        init_params['device'] = device
    task_params['checkpoint'] = cfg['checkpoint']
    if is_ddp:
        generator = get_seeded_generator(init_params['seed'])
    else:
        generator = get_seeded_generator(cfg['seed'])
    init_params['is_ddp'] = is_ddp
    if task == 'train' or task == 'train_test': # for train and validation
        task_params['loss'] = get_loss()
        train_dataset, collate = get_dataset(cfg['train_dataset']['name'], cfg['train_dataset']['params'], method='train')
        if is_ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=cfg['seed'], rank=rank, num_replicas=dist.get_world_size())
            per_gpu_batch_size = cfg['train_loader']['batch_size'] // dist.get_world_size()
            train_loader_cfg = cfg['train_loader'].copy()
            train_loader_cfg['batch_size'] = per_gpu_batch_size
            train_loader_cfg['shuffle'] = False
            task_params['train_dataloader'] = DataLoader(train_dataset, collate_fn=collate, sampler=train_sampler, **train_loader_cfg, worker_init_fn=seed_worker, generator=generator)
        else:
            task_params['train_dataloader'] = DataLoader(train_dataset, collate_fn=collate, **cfg['train_loader'], worker_init_fn=seed_worker, generator=generator)
        val_dataset, _ = get_dataset(cfg['val_dataset']['name'], cfg['val_dataset']['params'], method='val')
        task_params['val_dataloader'] = DataLoader(val_dataset, **cfg['val_loader'])
        task_params['optimizer'] = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'])
        task_params['scheduler'] = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], task_params['optimizer'])
        task_params['num_epochs'] = cfg['num_epochs']
    if task != 'train': # for test
        test_dataset, _ = get_dataset(cfg['test_dataset']['name'], cfg['test_dataset']['params'], method='test')
        task_params['test_dataloader'] = DataLoader(test_dataset, **cfg['test_loader'])
    return init_params, task_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sta_train.yaml')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'vis'])
    parser.add_argument('--is_ddp', action='store_true')
    args = parser.parse_args()

    init_params, task_params = load_config(args.config, args.task, args.is_ddp)
    trainer = DGTrainer(**init_params)
    if args.task == 'train':
        trainer.train(**task_params)
        if args.is_ddp:
            dist.destroy_process_group()
    elif args.task == 'test':
        trainer.test(**task_params)
    elif args.task == 'vis':
        trainer.vis(**task_params)
    else:
        print('This task does not exist')
        raise NotImplementedError