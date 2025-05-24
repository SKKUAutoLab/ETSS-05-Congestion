import numpy as np
import torch
from functools import partial
try:
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids
    from pykeops.torch.cluster import sort_clusters, from_matrix, swap_axes
    keops_available = True
except:
    keops_available = False
from .utils import squared_distances, distances
from .sinkhorn_divergence import scaling_parameters
from .sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop

cost_routines = {1 : (lambda x,y : distances(x,y)), 2 : (lambda x,y : squared_distances(x,y) / 2.)}

def softmin_tensorized(ε, C, f):
    B = C.shape[0]
    return - ε * ( f.view(B,1,-1) - C/ε ).logsumexp(2).view(B, -1, 1)

def sinkhorn_tensorized(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, debias = True, potentials = False, **kwargs):
    _, M, _ = y.shape
    if cost is None:
        cost = cost_routines[p]
    C_xx, C_yy = ( cost( x, x.detach()), cost( y, y.detach()) ) if debias else (None, None)
    C_xy, C_yx = ( cost( x, y.detach()), cost( y, x.detach()) )
    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )
    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin_tensorized, log_weights(α), log_weights(β), C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=debias )
    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=debias, potentials=potentials)

cost_formulas = {1 : "Norm2(X-Y)", 2 : "(SqDist(X,Y) / IntCst(2))"}

def softmin_online(ε, C_xy, f_y, log_conv=None):
    x, y = C_xy
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)

def keops_lse(cost, D, dtype="float32"):
    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )", "A = Vi(1)", "X = Vi({})".format(D), "Y = Vj({})".format(D), "B = Vj(1)", "P = Pm(1)", dtype = dtype)
    return log_conv

def sinkhorn_online(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None,  debias = True, potentials = False, **kwargs):
    N, D = x.shape
    M, _ = y.shape
    if cost is None:
        cost = cost_formulas[p]
    softmin = partial(softmin_online, log_conv=keops_lse(cost, D, dtype=str(x.dtype)[6:]))
    C_xx, C_yy = ( (x, x.detach()), (y, y.detach()) ) if debias else (None, None)
    C_xy, C_yx = ( (x, y.detach()), (y, x.detach()) )
    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )
    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin, log_weights(α), log_weights(β), C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=debias )
    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, debias=debias, potentials=potentials)

def softmin_multiscale(ε, C_xy, f_y, log_conv=None):
    x, y, ranges_x, ranges_y, ranges_xy = C_xy
    print(20*'#')
    print(x.shape, y.shape)
    if ranges_xy is not None:
        for i in range(len(ranges_xy)):
            print(ranges_xy[i].shape)
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x), ranges=ranges_xy ).view(-1)

def clusterize(α, x, scale=None, labels=None) :
    perm = None
    if labels is None and scale is None :
        return [α], [x], []
    else :
        x_lab = grid_cluster(x, scale) if labels is None else labels
        ranges_x, x_c, α_c = cluster_ranges_centroids(x, x_lab, weights=α)
        x_labels, perm = torch.sort(x_lab.view(-1))
        α, x = α[perm], x[perm]
        return [α_c, α], [x_c, x], [ranges_x], perm

def kernel_truncation( C_xy, C_yx, C_xy_, C_yx_, b_x, a_y, ε, truncate=None, cost=None, verbose=False):
    if truncate is None:
        return C_xy_, C_yx_
    else:
        x,  yd,   ranges_x,  ranges_y, _ = C_xy
        y,  xd,          _,         _, _ = C_yx
        x_, yd_, ranges_x_, ranges_y_, _ = C_xy_
        y_, xd_,         _,         _, _ = C_yx_
        with torch.no_grad():
            C      = cost(x, y)
            keep   = b_x.view(-1,1) + a_y.view(1,-1) > C - truncate*ε
            ranges_xy_ = from_matrix(ranges_x, ranges_y, keep)
            if verbose:
                ks, Cs = keep.sum(), C.shape[0]*C.shape[1]
                print("Keep {}/{} = {:2.1f}% of the coarse cost matrix.".format(ks, Cs, 100*float(ks) / Cs ) )
        return (x_, yd_, ranges_x_, ranges_y_, ranges_xy_), (y_, xd_, ranges_y_, ranges_x_, swap_axes(ranges_xy_))

def extrapolate_samples( b_x, a_y, ε, λ, C_xy, β_log, C_xy_, softmin=None ):
    yd = C_xy[1]
    x_ = C_xy_[0]
    C = (x_, yd, None, None, None)
    return λ * softmin(ε, C, (β_log + a_y/ε).detach() )

def sinkhorn_multiscale(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, truncate=5, cost=None, cluster_scale=None, debias = True, potentials = False, labels_x = None, labels_y = None, verbose=False, **kwargs):
    N, D = x.shape
    M, _ = y.shape
    if cost is None:
        cost = cost_formulas[p], cost_routines[p]
    cost_formula, cost_routine = cost[0], cost[1]
    softmin = partial(softmin_multiscale, log_conv=keops_lse(cost_formula, D, dtype=str(x.dtype)[6:])) 
    extrapolate = partial(extrapolate_samples, softmin=softmin)
    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )
    if cluster_scale is None:
        cluster_scale = diameter / (np.sqrt(D) * 2000**(1/D))
    [α_c, α], [x_c, x], [ranges_x], perm_x = clusterize(α, x, scale=cluster_scale, labels=labels_x)
    [β_c, β], [y_c, y], [ranges_y], perm_y = clusterize(β, y, scale=cluster_scale, labels=labels_y)
    jumps = [ len(ε_s)-1 ]
    for i, ε in enumerate(ε_s[2:]):
        if cluster_scale**p > ε:
            jumps = [i+1]
            break
    if verbose: 
        print("{}x{} clusters, computed at scale = {:2.3f}".format(len(x_c), len(y_c), cluster_scale))
        print("Successive scales : ", ', '.join(["{:.3f}".format(x**(1/p)) for x in ε_s]))
        if jumps[0] >= len(ε_s)-1:
            print("Extrapolate from coarse to fine after the last iteration.")
        else:
            print("Jump from coarse to fine between indices {} (σ={:2.3f}) and {} (σ={:2.3f}).".format(jumps[0], ε_s[jumps[0]]**(1/p), jumps[0]+1, ε_s[jumps[0]+1]**(1/p)))
    α_logs = [ log_weights(α_c), log_weights(α) ]
    β_logs = [ log_weights(β_c), log_weights(β) ]
    C_xxs = [ (x_c, x_c.detach(), ranges_x, ranges_x, None), (  x,   x.detach(),     None,     None, None) ] if debias else None
    C_yys = [ (y_c, y_c.detach(), ranges_y, ranges_y, None), (  y,   y.detach(),     None,     None, None) ] if debias else None
    C_xys = [ (x_c, y_c.detach(), ranges_x, ranges_y, None), (  x,   y.detach(),     None,     None, None) ]
    C_yxs = [ (y_c, x_c.detach(), ranges_y, ranges_x, None), (  y,   x.detach(),     None,     None, None) ]
    a_x, b_y, a_y, b_x = sinkhorn_loop(softmin, α_logs, β_logs, C_xxs, C_yys, C_xys, C_yxs, ε_s, ρ, jumps=jumps, cost=cost_routine, kernel_truncation=partial(kernel_truncation, verbose=verbose), truncate=truncate, extrapolate=extrapolate, debias = debias)
    cost = sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, debias=debias, potentials=potentials)
    if potentials:
        F_x, G_y = cost
        f_x, g_y = F_x.clone(), G_y.clone()
        f_x[perm_x], g_y[perm_y] = F_x, G_y
        return f_x, g_y
    else:
        return cost