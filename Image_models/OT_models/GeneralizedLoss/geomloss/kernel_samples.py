import numpy as np
import torch
try:
    from pykeops.torch import generic_sum
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, sort_clusters, from_matrix, swap_axes
    keops_available = True
except:
    keops_available = False
from .utils import scal, squared_distances, distances

class DoubleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return 2*grad_output

def double_grad(x):
    return DoubleGrad.apply(x)

def gaussian_kernel(x, y, blur=.05):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def laplacian_kernel(x, y, blur=.05):
    C = distances(x / blur, y / blur)
    return (- C ).exp()

def energy_kernel(x, y, blur=None):
    return - distances(x, y)

kernel_routines = {"gaussian" : gaussian_kernel, "laplacian": laplacian_kernel, "energy"   : energy_kernel}

def kernel_tensorized(α, x, β, y, blur=.05, kernel=None, name=None, potentials=False,**kwargs):
    _, M, _ = y.shape
    if kernel is None:
        kernel = kernel_routines[name]
    K_xx = kernel( double_grad(x), x.detach(), blur=blur)
    K_yy = kernel( double_grad(y), y.detach(), blur=blur)
    K_xy = kernel( x, y, blur=blur)
    a_x = torch.matmul( K_xx, α.detach().unsqueeze(-1) ).squeeze(-1)
    b_y = torch.matmul( K_yy, β.detach().unsqueeze(-1) ).squeeze(-1)
    b_x = torch.matmul( K_xy, β.unsqueeze(-1)          ).squeeze(-1)
    if potentials:
        a_y = torch.matmul( K_xy.transpose(1,2), α.unsqueeze(-1)).squeeze(-1)
        return a_x - b_x, b_y - a_y
    else:
        return .5 * (double_grad(α) * a_x).sum(1) + .5 * (double_grad(β) * b_y).sum(1) -  (α * b_x).sum(1)

kernel_formulas = {"gaussian" : ("Exp(-SqDist(X,Y) / IntCst(2))", True ), "laplacian": ("Exp(-Norm2(X-Y))",   True ), "energy"   : ("(-Norm2(X-Y))",      False)}

def kernel_keops(kernel, α, x, β, y, potentials=False, ranges_xx = None, ranges_yy = None, ranges_xy = None):
    D = x.shape[1]
    kernel_conv = generic_sum( "(" + kernel + " * B)", "A = Vi(1)", "X = Vi({})".format(D), "Y = Vj({})".format(D), "B = Vj(1)" )
    a_x = kernel_conv(double_grad(x), x.detach(), α.detach().view(-1,1), ranges=ranges_xx)
    b_y = kernel_conv(double_grad(y), y.detach(), β.detach().view(-1,1), ranges=ranges_yy)
    b_x = kernel_conv(x, y, β.view(-1,1), ranges=ranges_xy)
    if potentials:
        a_y = kernel_conv(y, x, α.view(-1,1), ranges=swap_axes(ranges_xy))
        return a_x - b_x, b_y - a_y
    else:
        return .5 * scal( double_grad(α), a_x ) + .5 * scal( double_grad(β), b_y )  -  scal( α, b_x )

def kernel_preprocess(kernel, name, x, y, blur):
    if not keops_available:
        raise ImportError("The 'pykeops' library could not be loaded: " + "'online' and 'multiscale' backends are not available.")
    if kernel is None:
        kernel, rescale = kernel_formulas[name]
    else:
        rescale = True
    center = (x.mean(0, keepdim=True) + y.mean(0,  keepdim=True)) / 2
    x, y = x - center, y - center
    if rescale :
        x, y = x / blur, y / blur
    return kernel, x, y

def kernel_online(α, x, β, y, blur=.05, kernel=None, name=None, potentials=False, **kwargs):
    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)
    return kernel_keops(kernel, α, x, β, y, potentials=potentials)

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs-mins).norm().item()
    return diameter

def kernel_multiscale(α, x, β, y, blur=.05, kernel=None, name=None, truncate=5, diameter=None, cluster_scale=None, potentials=False, verbose=False, **kwargs):
    if truncate is None or name == "energy":
        return kernel_online( α, x, β, y, blur=blur, kernel=kernel, truncate=truncate, name=name, potentials=potentials, **kwargs )
    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)
    if cluster_scale is None: 
        D = x.shape[-1]
        if diameter is None:
            diameter = max_diameter(x.view(-1,D), y.view(-1,D))
        else:
            diameter = diameter / blur
        cluster_scale = diameter / (np.sqrt(D) * 2000**(1/D))
    cell_diameter = cluster_scale * np.sqrt( x.shape[1] )
    x_lab = grid_cluster(x, cluster_scale) 
    y_lab = grid_cluster(y, cluster_scale)
    ranges_x, x_c, α_c = cluster_ranges_centroids(x, x_lab, weights=α)
    ranges_y, y_c, β_c = cluster_ranges_centroids(y, y_lab, weights=β)
    if verbose: 
        print("{}x{} clusters, computed at scale = {:2.3f}".format(len(x_c), len(y_c), cluster_scale))
    (α, x), x_lab = sort_clusters( (α, x), x_lab)
    (β, y), y_lab = sort_clusters( (β, y), y_lab)
    with torch.no_grad():
        C_xx = squared_distances( x_c, x_c)
        C_yy = squared_distances( y_c, y_c)
        C_xy = squared_distances( x_c, y_c)
        keep_xx = ( C_xx <= (truncate + cell_diameter)**2 )
        keep_yy = ( C_yy <= (truncate + cell_diameter)**2 )
        keep_xy = ( C_xy <= (truncate + cell_diameter)**2 )
        ranges_xx = from_matrix(ranges_x, ranges_x, keep_xx)
        ranges_yy = from_matrix(ranges_y, ranges_y, keep_yy)
        ranges_xy = from_matrix(ranges_x, ranges_y, keep_xy)
    return kernel_keops(kernel, α, x, β, y, potentials=potentials, ranges_xx=ranges_xx, ranges_yy=ranges_yy, ranges_xy=ranges_xy)