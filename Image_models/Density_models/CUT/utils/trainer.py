import os
from utils.logger import logger

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger(os.path.join(args.output_dir, 'train.log'))

    def setup(self):
        pass

    def train(self):
        pass