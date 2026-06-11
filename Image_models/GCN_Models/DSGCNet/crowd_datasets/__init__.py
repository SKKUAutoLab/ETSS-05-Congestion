def build_dataset(args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
    else:
        print('This dataset does not exist')
        raise NotImplementedError