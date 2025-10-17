import torch

def count_merge_low2high_batch(clow,chigh):
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (clow.device.type == 'cuda')
    rate = int(chigh.size()[-1] / clow.size()[-1])
    norm = 1 / (float(rate)**2)
    cl2h = torch.zeros(chigh.size())
    if IF_gpu:
        clow, chigh, cl2h = clow.cuda(), chigh.cuda(), cl2h.cuda()
    for rx in range(rate):
        for ry in range(rate):
            cl2h[:, :, rx::rate, ry::rate] = clow * norm
    if not IF_ret_gpu:
        cl2h = cl2h.cpu()
    return cl2h