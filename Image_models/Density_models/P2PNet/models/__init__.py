from .p2pnet import build

def build_model(args, training=False):
    return build(args, training)