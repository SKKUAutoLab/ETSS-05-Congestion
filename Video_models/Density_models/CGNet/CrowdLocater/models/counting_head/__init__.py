from .HRSegment import build_counting_head as HRSegment

def build_counting_head(args):
    if args.name == "HRSegMent":
        return HRSegment(args)
    else:
        print('This model does not exist')
        raise NotImplementedError