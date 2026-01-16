from .SHA import build as build_sha

def build_dataset(image_set, args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        return build_sha(image_set, args)
    else:
        print('This dataset does not exist')
        raise NotImplementedError