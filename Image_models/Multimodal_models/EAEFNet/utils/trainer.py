import os

class Trainer(object):
    def __init__(self, args):
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.args = args

    def setup(self):
        pass

    def train(self):
        pass