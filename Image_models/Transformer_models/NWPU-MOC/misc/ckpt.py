import io
import os
import os.path as osp
import pkgutil
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
import torch
import torchvision
from torch.utils import model_zoo
from torch.nn import functional as F
import mmcv
from mmcv.fileio import FileClient
from mmcv.fileio import load as load_file
from mmcv.parallel import is_module_wrapper
from mmcv.utils import mkdir_or_exist
from mmcv.runner import get_dist_info

ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

def _get_mmcv_home():
    mmcv_home = os.path.expanduser(os.getenv(ENV_MMCV_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))
    mkdir_or_exist(mmcv_home)
    return mmcv_home

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def load_url_dist(url, model_dir=None):
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint

def load_pavimodel_dist(model_path, map_location=None):
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError('Please install pavi to load checkpoint from modelcloud.')
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            downloaded_file = osp.join(tmp_dir, model.name)
            model.download(downloaded_file)
            checkpoint = torch.load(downloaded_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                downloaded_file = osp.join(tmp_dir, model.name)
                model.download(downloaded_file)
                checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint

def load_fileclient_dist(filename, backend, map_location):
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint

def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls

def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)
    return default_urls

def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)
    return mmcls_urls

def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0], 'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)
    return deprecate_urls

def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)
    return new_checkpoint

def _load_checkpoint(filename, map_location=None):
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
                          f'of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        checkpoint = load_fileclient_dist(
            filename, backend='ceph', map_location=map_location)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H*W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        if table_key not in model.state_dict():
            print('missing keys: ' + table_key)
            continue
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint