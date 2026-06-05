from models.seg_att_prop_vgg import SegAttPropVGG

class Model():
    def __init__(self, args):
        if 'vgg' in args.model:
            self.model = SegAttPropVGG(args)
        else:
            print('This model does not exist')
            raise NotImplementedError

    def get_model(self):
        return self.model