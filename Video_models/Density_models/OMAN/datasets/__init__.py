from datasets.Sense_dataset import build_dataset as build_SENSE_dataset_train
from datasets.Sense_dataset import build_video_dataset as build_SENSE_dataset_test
from datasets.HT21_dataset import build_dataset as build_HT21_dataset_train
from datasets.HT21_dataset import build_video_dataset as build_HT21_dataset_test

def build_dataset(dataset_file, root, annotation_dir='', max_len=3000, train=False, step=15):
    if train:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_train(root, annotation_dir, max_len, train=train, step=step)
        elif dataset_file == 'HT21':
            return build_HT21_dataset_train(root, max_len, train=train, step=step)
        else:
            print('This dataset does not exist')
            raise NotImplementedError
    else:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_test(root, annotation_dir)
        elif dataset_file == 'HT21':
            return build_HT21_dataset_test(root)
        else:
            print('This dataset does not exist')
            raise NotImplementedError