import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from .broyden import broyden, broyden_batch, broyden_aligned

'''

    min loss_x(x,y)
    s.t. y = argmin loss_y(x,y)

'''


# input grads of x (tuple), x: list of tensors
def paddedlist(grads, x):
    list_grads = list(grads)
    for i in range(len(list_grads)):
        if list_grads[i] is None:
            list_grads[i] = torch.zeros_like(x[i])
    return list_grads


def implicit_gradient(x, y, lx, ly, n=10, alpha=0.4):
    v1 = grad(lx, y, retain_graph=True)[0]
    v = grad(ly, y, create_graph=True)[0]  # this must use create graph, others can be retain_graph

    v2 = approxInversHVP(v1, v, y, n, alpha)
    v3 = grad(v, x, grad_outputs=v2, retain_graph=True)[0]

    gx = grad(lx, x, allow_unused=True)[0]
    if gx is None:
        gx = torch.zeros_like(x)

    vx = gx - v3
    return vx


'''
    Computer v \dot (\partial f / \partial w)^-1

    notice |H|<1 for convergence

'''


def approxInversHVP(v, f, w, n_iter=10, alpha=0.1):
    p = v
    for i in range(n_iter):
        z = grad(f, w, grad_outputs=v, retain_graph=True)[0]

        # + cannot replace with += or -=, or use v.clone()
        v = v - alpha * z
        p = p + v
    return alpha * p


'''

    min loss_x(x,y)
    s.t. y = argmin loss_y(x,y)

    where x is a single tensor, y is a list of tensors in a NN (by list(net.parameters()))
'''


def implicit_gradient_neumann_net(x, y, lx, ly, n=10, alpha=0.4):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True), y)  # derivative for a list of tensors
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    v2 = approxInversHVP_net(v1, v, y, n, alpha)
    v3 = grad(v, x, grad_outputs=v2, retain_graph=True)[0]  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)[0]
    if gx is None:
        gx = torch.zeros_like(x)

    vx = gx - v3
    return vx


'''
    input: x: list of parameters of a nn
           y: list of parameters of a nn
'''


def implicit_gradient_neumann_net2(x, y, lx, ly, n=10, alpha=0.4):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True),
                    y)  # derivative for a list of tensors, it might be None because it might involves derivatives of parameters which are 0
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    v2 = approxInversHVP_net(v1, v, y, n, alpha)  # TODO: check why this  explodes
    v3 = paddedlist(grad(v, x, grad_outputs=v2, retain_graph=True, allow_unused=True),
                    x)  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)
    if gx[0] is None:
        gx = [torch.zeros_like(_x) for _x in x]
    else:
        gx = list(gx)

    vx = [gx[i] - v3[i] for i in range(len(gx))]
    return vx


'''

    min loss_x(x,y)
    s.t. y = argmin loss_y(x,y)

    where x is a single tensor, y is a list of tensors in a NN (by list(net.parameters()))
'''


def implicit_gradient_t1t2_net(x, y, lx, ly):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True), y)  # derivative for a list of tensors
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    v3 = grad(v, x, grad_outputs=v1, retain_graph=True)[0]  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)[0]
    if gx is None:
        gx = torch.zeros_like(x)

    vx = gx - v3
    return vx


'''
    input: x: list of parameters of a nn
           y: list of parameters of a nn
'''


def implicit_gradient_t1t2_net2(x, y, lx, ly):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True),
                    y)  # derivative for a list of tensors, it might be None because it might involves derivatives of parameters which are 0
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    v3 = paddedlist(grad(v, x, grad_outputs=v1, retain_graph=True, allow_unused=True),
                    x)  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)
    if gx[0] is None:
        gx = [torch.zeros_like(_x) for _x in x]
    else:
        gx = list(gx)

    vx = [gx[i] - v3[i] for i in range(len(gx))]
    return vx


'''

    min loss_x(x,y)
    s.t. y = argmin loss_y(x,y)

    where x is a single tensor, y is a list of tensors in a NN (by list(net.parameters()))
'''


def implicit_gradient_trad_net(x, y, lx, ly):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True), y)  # derivative for a list of tensors, \J / \w
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph  ,\E / \w

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    v3 = grad(v, x, grad_outputs=v1, retain_graph=True)[0]  # derivative for a single tensor, \ v/ \theta

    gx = grad(lx, x, allow_unused=True)[0]  # \J / \theta
    if gx is None:
        gx = torch.zeros_like(x)

    vx = gx - v3
    return vx


'''
    input: x: list of parameters of a nn
           y: list of parameters of a nn
'''


def implicit_gradient_trad_net2(x, y, lx, ly):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True),
                    y)  # derivative for a list of tensors, it might be None because it might involves derivatives of parameters which are 0
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    v3 = paddedlist(grad(v, x, grad_outputs=v1, retain_graph=True, allow_unused=True),
                    x)  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)
    if gx[0] is None:
        gx = [torch.zeros_like(_x) for _x in x]
    else:
        gx = list(gx)

    vx = [gx[i] - v3[i] for i in range(len(gx))]
    return vx


'''
    Computer v \dot (\partial f / \partial w)^-1

    notice |H|<1 for convergence

'''


def approxInversHVP_net(v, f, w, n_iter=10, alpha=0.1):
    p = [v_.clone() for v_ in v]
    for i in range(n_iter):
        z = []
        for j in range(len(v)):
            z.extend(grad(f[j], w[j], grad_outputs=v[j], retain_graph=True))

            # + cannot replace with += or -=, or use v.clone()
            v[j] = v[j] - alpha * z[-1]
            p[j] = p[j] + v[j]

        # print(p[0])
    p = [alpha * p_ for p_ in p]
    return p


def implicit_gradient_broyden_net(x, y, lx, ly, threshold=30, max_iter=-1, eps=1e-6, ls=True, beta=1e-3):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True),
                    y)  # derivative for a list of tensors, it might be None because it might involves derivatives of parameters which are 0
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    # v2 = broyden_net(v1, v, y, max_rank=threshold, eps=eps, ls=ls,beta=beta)
    # v2 = broyden_net_ls(v1, v, y, max_rank=threshold, ls=ls, eps=eps,beta=0.01)
    # v2 = broyden_net_cached(v1, v, y, max_rank=threshold, ls=ls, eps=eps,beta=0.001)
    v2 = broyden_all(v1, v, y, max_rank=threshold, max_iter=max_iter, ls=ls, eps=eps, beta=0.01)
    v3 = grad(v, x, grad_outputs=v2, retain_graph=True)[0]  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)[0]
    if gx is None:
        gx = torch.zeros_like(x)

    vx = gx - v3
    return vx


def implicit_gradient_broyden_net2(x, y, lx, ly, threshold=30, max_iter=-1, eps=1e-6, ls=True, beta=1e-3):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True),
                    y)  # derivative for a list of tensors, it might be None because it might involves derivatives of parameters which are 0
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    # v2 = broyden_net(v1, v, y, max_rank=threshold,eps=eps,ls=ls,beta=beta)
    v2 = broyden_all(v1, v, y, max_rank=threshold, max_iter=max_iter, ls=ls, eps=eps, beta=0.01)
    # v2 = approxInversHVP_net(v1, v, y, n, alpha)  # TODO: check why this  explodes
    v3 = paddedlist(grad(v, x, grad_outputs=v2, retain_graph=True, allow_unused=True),
                    x)  # derivative for a single tensor

    gx = grad(lx, x, allow_unused=True)
    if gx[0] is None:
        gx = [torch.zeros_like(_x) for _x in x]
    else:
        gx = list(gx)

    vx = [gx[i] - v3[i] for i in range(len(gx))]
    return vx


# calculate v \f /\w^-1
def broyden_net(v, f, w, max_rank=30, eps=1e-6, p0=None, ls=True, beta=0.001):
    p = [torch.zeros_like(v_) for v_ in v]
    for j in range(len(p)):
        g_v = lambda x: beta * grad(f[j], w[j], grad_outputs=x, retain_graph=True)[0] - v[j]  # could be [n,m], or [n]
        # broyden_result = broyden(g_v, p[j], max_rank, eps=eps, ls=ls)
        broyden_result = broyden_aligned(g_v, p[j], max_rank, eps=eps, ls=ls, verbose=False)
        p[j], status = broyden_result['result'] * beta, broyden_result['prot_break']
        print('{} @ diff initial {} final {} Opt Ratio {}'.format(j, broyden_result['trace'][0], broyden_result['diff'],
                                                                  broyden_result['trace'][0] / (
                                                                          broyden_result['diff'] + 1e-9)))
        if status:
            # collect_total_jacobian(f[j], w[j], v[j], p[j])
            print('Failed @ {}'.format(j))
            # exit()
    return p


# calculate v \f /\w^-1
def broyden_net_ls(v, f, w, max_rank=30, eps=1e-6, p0=None, ls=True, beta=0.01):
    p = [v_.clone().detach() for v_ in v]
    # p = [torch.zeros_like(v_) for v_ in v]
    for j in range(len(p)):

        v_j_norm = v[j].norm() + 1e-7
        p[j] = p[j] / v_j_norm

        # g_v = lambda x: grad(f[j], w[j], grad_outputs=x,retain_graph=True)[0]-v[j] / v_j_norm  #could be [n,m], or [n]
        g_v = lambda x: 0.01 * \
                        grad(f[j], w[j], grad_outputs=grad(f[j], w[j], grad_outputs=x, retain_graph=True)[0] - v[j],
                             retain_graph=True)[0] + beta * x  # could be [n,m], or [n]
        # broyden_result = broyden(g_v, p[j], max_rank, eps=eps, ls=ls)
        broyden_result = broyden_aligned(g_v, p[j], max_rank, eps=eps, ls=ls, verbose=False)
        p[j], status = broyden_result['result'], broyden_result['prot_break']

        p[j] = p[j] * v_j_norm * 0.01
        print('{} @ diff initial {} final {} Opt Ratio {}'.format(j, broyden_result['trace'][0], broyden_result['diff'],
                                                                  broyden_result['trace'][0] / (
                                                                          broyden_result['diff'] + 1e-9)))
        if status or (broyden_result['trace'][0] / (broyden_result['diff'] + 1e-9) < 5 and np.abs(
                broyden_result['trace'][0]) > 1e-3):
            # collect_total_jacobian(f[j], w[j], v[j], p[j])
            print('Failed @ {}'.format(j))
            # exit()
    return p


### in progress and testing
previous_p = []


# calculate v \f /\w^-1
def broyden_net_cached(v, f, w, max_rank=30, eps=1e-6, p0=None, ls=True, beta=0.001):
    global previous_p
    if len(previous_p) == 0:  # initialize
        p = [v_.detach().clone() / (v_.norm() + 1e-8) for v_ in v]
    else:
        p = [p_.clone().detach() for p_ in previous_p]
    v_norms = []
    for j in range(len(p)):

        v_j_norm = v[j].norm()
        # p[j] = p[j] / v_j_norm

        g_v = lambda x: grad(f[j], w[j], grad_outputs=x, retain_graph=True)[0] - v[j] / (
                v_j_norm + 1e-9)  # could be [n,m], or [n]
        # g_v = lambda x: grad(f[j], w[j], grad_outputs=grad(f[j], w[j], grad_outputs=x,retain_graph=True)[0]-v[j],retain_graph=True)[0]+beta*x   #could be [n,m], or [n]
        # broyden_result = broyden(g_v, p[j], max_rank, eps=eps, ls=ls)
        broyden_result = broyden_aligned(g_v, p[j], max_rank, eps=eps, ls=ls, verbose=False)
        p[j], status = broyden_result['result'], broyden_result['prot_break']

        p[j] = p[j] * v_j_norm
        v_norms.append(v_j_norm)
        # cache p

        print('{} @ diff initial {} final {} Opt Ratio {}'.format(j, broyden_result['trace'][0], broyden_result['diff'],
                                                                  broyden_result['trace'][0] / (
                                                                          broyden_result['diff'] + 1e-9)))
        if status or (broyden_result['trace'][0] / (broyden_result['diff'] + 1e-9) < 5 and np.abs(
                broyden_result['trace'][0]) > 1e-3):
            collect_total_jacobian(f[j], w[j], v[j], p[j])
            print('Failed @ {}'.format(j))
            exit()

    previous_p = [None] * len(v_norms)
    for k in range(len(previous_p)):
        # previous_p[k] = p[k].detach() / (v_norms[k]+1e-8)
        previous_p[k] = torch.randn_like(p[k])
        previous_p[k] = previous_p[k] / previous_p[k].norm()

    return p


# f = \E / \ w_i,

'''
    Steps determined by max rank, if max iter is -1, else by max iter
'''


def broyden_all(v, f, w, max_rank=30, max_iter=-1, eps=1e-6, p0=None, ls=True, beta=0.01):
    v_cat = torch.cat([v_.flatten() for v_ in v])
    param_shapes = [v_.shape for v_ in v]  # n* shape
    param_lens = torch.tensor([v_.numel() for v_ in v]).long()  # n
    param_idxes = torch.cat([torch.zeros([1]), torch.cumsum(param_lens, dim=0)]).long()  # 1+n

    v_norm = v_cat.norm() + 1e-8
    v_cat = v_cat / v_norm
    p_cat = v_cat.clone()

    def mixed_jacobian_vjp(x):
        out_grad_vec = torch.zeros_like(x)
        E = torch.zeros([1])
        for i in range(len(v)):
            x_i = x[param_idxes[i]:param_idxes[i + 1]].view(param_shapes[i])
            E += (x_i * f[i]).sum()

        for i in range(len(v)):
            out_grad_vec[param_idxes[i]: param_idxes[i + 1]] = (grad(E, w[i], retain_graph=True)[0]).flatten()

        out_grad_vec = out_grad_vec - v_cat
        return out_grad_vec

    broyden_result = broyden_aligned(mixed_jacobian_vjp, p_cat, max_rank, steps=max_iter, eps=eps, ls=ls, verbose=False)
    p_cat = broyden_result['result']
    p_cat = p_cat * v_norm
    print('Diff initial {} final {} Opt Ratio {}'.format(broyden_result['trace'][0], broyden_result['diff'],
                                                         broyden_result['trace'][0] / (broyden_result['diff'] + 1e-9)))
    # packed to tensors
    p = []
    for i in range(len(v)):
        p.append(p_cat[param_idxes[i]: param_idxes[i + 1]].view(param_shapes[i]))
    return p


def implicit_adjoint_hessian_product_broyden(x, y, lx, ly, threshold=30, eps=1e-6, ls=True, beta=1e-3):
    v1 = list(grad(lx, y, retain_graph=True))  # derivative for a list of tensors
    v = list(grad(ly, y, create_graph=True))  # this must use create graph, others can be retain_graph

    # v2 = approxInversHVP_net(v1, v, y, n, alpha)
    # v2 = broyden_net(v1, v, y, max_rank=threshold, eps=eps, ls=ls,beta=1e-3)
    v2 = broyden_all(v1, v, y, max_rank=threshold, eps=eps, ls=ls, beta=1e-3)

    return v2


def implicit_adjoint_hessian_product_neumann(x, y, lx, ly, n=10, alpha=0.4):
    v1 = paddedlist(grad(lx, y, retain_graph=True, allow_unused=True), y)  # derivative for a list of tensors
    v = paddedlist(grad(ly, y, create_graph=True, allow_unused=True),
                   y)  # this must use create graph, others can be retain_graph

    v2 = approxInversHVP_net(v1, v, y, n, alpha)

    return v2


## save \f/\w and v

def collect_total_jacobian(f, w, v, p):
    f_ = lambda x: grad(f, w, grad_outputs=x, retain_graph=True)[0]

    e = torch.zeros_like(v).flatten()
    n_dim = e.shape[0]
    Jacobian = torch.zeros(n_dim, n_dim)
    for i in range(e.shape[0]):
        ei = e.clone()
        ei[i] = 1
        ei = ei.view_as(v)
        gradi = f_(ei)
        Jacobian[i] = gradi.flatten()
    Jacobian = Jacobian.detach().cpu().numpy()
    np.savetxt('./data/jacobian_test.txt', Jacobian)
    np.savetxt('./data/rightarr.txt', v.detach().flatten().cpu().numpy())
