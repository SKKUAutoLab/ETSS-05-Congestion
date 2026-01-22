import os

class Trainer(object):
    def __init__(self, args):
        self.save_dir = args.output_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.args = args

    def setup(self):
        pass

    def train(self):
        pass