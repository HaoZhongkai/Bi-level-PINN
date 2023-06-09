import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored
import logging

logger = logging.getLogger()


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1 * alpha2 * derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def brmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bi, bij -> bj', x, part_Us)  # (N, threshold)
    return -x + torch.einsum('bj, bji -> bi', xTU, part_VTs)  # (N, d)


def bmatvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bji, bi -> bj', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bij, bj -> bi', part_Us, VTx)  # (N, d)


def broyden_batch(g_, x0, threshold, eps, ls=False, name="unknown"):
    # LBFGS_thres = min(threshold, 20)
    LBFGS_thres = threshold

    x0_shape = x0.shape
    x0 = x0.view(x0_shape[0], -1)

    bsz, total_hsize = x0.size()
    eps = eps * np.sqrt(np.prod(x0.shape))

    def g(x):
        return g_(x.view(x0_shape)).view(bsz, -1)

    x_est = x0  # (bsz, d)
    gx = g(x_est)  # (bsz, d)
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, LBFGS_thres).to(x0)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize).to(x0)
    update = -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]

    # To be used in protective breaks
    protect_thres = 1e6
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    while new_objective >= eps and nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite + 1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3 * eps and nstep == threshold and np.max(trace[-threshold:]) / np.min(
                trace[-threshold:]) < 1.3:
            logger.info('Iterations exceeded 30 for broyden')
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            logger.info('Broyden failed')
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :(nstep - 1) % LBFGS_thres], VTs[:, :(nstep - 1) % LBFGS_thres]
        vT = brmatvec(part_Us, part_VTs, delta_x)  # (N, d)
        u = (delta_x - bmatvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bi, bi -> b', vT, delta_gx)[:, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, (nstep - 1) % LBFGS_thres] = vT
        Us[:, :, (nstep - 1) % LBFGS_thres] = u
        update = -bmatvec(Us[:, :, :nstep], VTs[:, :nstep], gx)

    Us, VTs = None, None
    return {"result": lowest_xest.view(x0_shape),
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold}


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    # xTU = torch.einsum('bi, bij -> bj', x, part_Us) # (N, threshold)
    # return -x + torch.einsum('bj, bji -> bi', xTU, part_VTs) # (N, d)
    xTU = x @ part_Us
    return -x + xTU @ part_VTs


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    # VTx = torch.einsum('bji, bi -> bj', part_VTs, x) # (N, threshold)
    # return -x + torch.einsum('bij, bj -> bi', part_Us, VTx) # (N, d)
    VTx = part_VTs @ x
    return -x + part_Us @ VTx


'''
    broyden without batch for network weight
    x0: [n*m] or [n]
    g_(x0) : [n*m] or [n]
'''


def broyden(g_, x0, threshold, eps, ls=False, name="unknown"):
    # LBFGS_thres = min(threshold, 20)
    LBFGS_thres = threshold

    x0_shape = x0.shape
    x0 = x0.view(-1)

    total_hsize = x0.size(0)
    eps = eps * np.sqrt(np.prod(x0.shape))

    def g(x):
        return g_(x.view(x0_shape)).view(-1)

    x_est = x0  # (bsz, d)
    gx = g(x_est)  # (bsz, d)
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(total_hsize, LBFGS_thres).to(x0)
    VTs = torch.zeros(LBFGS_thres, total_hsize).to(x0)
    update = -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]

    # To be used in protective breaks
    protect_thres = 1e6
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    while new_objective >= eps and nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite + 1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            # print('Solution found')
            break
        if new_objective < 3 * eps and nstep == threshold and np.max(trace[-threshold:]) / np.min(
                trace[-threshold:]) < 1.3:
            # logger.info('Iterations exceeded 30 for broyden')
            print('Iterations exceeded 30 for broyden')
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            # logger.info('Broyden failed')
            print('Broyden failed')
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :(nstep - 1) % LBFGS_thres], VTs[:(nstep - 1) % LBFGS_thres]
        vT = rmatvec(part_Us, part_VTs, delta_x)  # (N, d)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / (vT @ delta_gx).unsqueeze(-1)
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[(nstep - 1) % LBFGS_thres] = vT
        Us[:, (nstep - 1) % LBFGS_thres] = u
        update = -matvec(Us[:, :nstep], VTs[:nstep], gx)

    Us, VTs = None, None
    return {"result": lowest_xest.view(x0_shape),
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold}


def rmatvec_aligned(part_Us, part_VTs, x, alpha):
    # Compute x^T(-I + UV^T)
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -alpha * x
    # xTU = torch.einsum('bi, bij -> bj', x, part_Us) # (N, threshold)
    # return -x + torch.einsum('bj, bji -> bi', xTU, part_VTs) # (N, d)
    xTU = x @ part_Us
    return -alpha * x + xTU @ part_VTs


def matvec_aligned(part_Us, part_VTs, x, alpha):
    # Compute (-I + UV^T)x
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -alpha * x
    # VTx = torch.einsum('bji, bi -> bj', part_VTs, x) # (N, threshold)
    # return -x + torch.einsum('bij, bj -> bi', part_Us, VTx) # (N, d)
    VTx = part_VTs @ x
    return -alpha * x + part_Us @ VTx


'''
    Some broyden method aligned with scipy
'''


def broyden_aligned(g_, x0, threshold, eps, steps=-1, verbose=False, ls=False, use_initializer=False, name="unknown"):
    # LBFGS_thres = min(threshold, 20)
    LBFGS_thres = threshold
    total_steps = LBFGS_thres if steps < 0 else steps
    x0_shape = x0.shape
    x0 = x0.view(-1)

    total_hsize = x0.size(0)
    eps = eps * np.sqrt(np.prod(x0.shape))

    def g(x):
        return g_(x.view(x0_shape)).view(-1)

    x_est = x0  # (bsz, d)
    gx = g(x_est)  # (bsz, d)
    nstep = 0
    tnstep = 0

    alpha_initial = 0.5 * max(torch.norm(x0), 1) / (torch.norm(gx))

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(total_hsize, LBFGS_thres).to(x0)
    VTs = torch.zeros(LBFGS_thres, total_hsize).to(x0)
    # update = -gx
    update = -matvec_aligned(Us[:, :nstep], VTs[:nstep], gx, alpha_initial)

    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]

    # To be used in protective breaks
    protect_thres = 1e6
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    while new_objective >= eps and nstep < total_steps:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)

        # TODO: check whether we need scaling

        # if use_initializer:
        #     alpha_initial = ((delta_x @ delta_gx) /( delta_gx @ delta_gx + 1e-8))

        nstep += 1
        tnstep += (ite + 1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        if verbose:
            print('Step {} |F(x)|: {}'.format(tnstep, torch.max(torch.abs(gx)).item()))
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            # print('Solution found')
            break
        if new_objective < 3 * eps and nstep == threshold and np.max(trace[-threshold:]) / np.min(
                trace[-threshold:]) < 1.3:
            # logger.info('Iterations exceeded 30 for broyden')
            print('Iterations exceeded 30 for broyden')
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            # logger.info('Broyden failed')
            print('Broyden failed')
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :(nstep - 1) % LBFGS_thres], VTs[:(nstep - 1) % LBFGS_thres]
        vT = rmatvec_aligned(part_Us, part_VTs, delta_x, alpha_initial)  # (N, d)
        # u = (delta_x - matvec_aligned(part_Us, part_VTs, delta_gx,alpha_initial)) / (vT @ delta_gx)
        u = (delta_x - matvec_aligned(part_Us, part_VTs, delta_gx, alpha_initial))
        vT = vT / (vT @ delta_gx)
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[(nstep - 1) % LBFGS_thres] = vT
        Us[:, (nstep - 1) % LBFGS_thres] = u
        update = -matvec_aligned(Us[:, :nstep], VTs[:nstep], gx, alpha_initial)
        # restart
        if ((nstep - 1) % LBFGS_thres) == 0:
            VTs.zero_()
            Us.zero_()

    Us, VTs = None, None
    return {"result": lowest_xest.view(x0_shape),
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold}


def analyze_broyden(res_info, err=None, judge=True, name='forward', training=True, save_err=True):
    """
    For debugging use only :-)
    """
    res_est = res_info['result']
    nstep = res_info['nstep']
    diff = res_info['diff']
    diff_detail = res_info['diff_detail']
    prot_break = res_info['prot_break']
    trace = res_info['trace']
    eps = res_info['eps']
    threshold = res_info['threshold']
    if judge:
        return nstep >= threshold or (nstep == 0 and (diff != diff or diff > eps)) or prot_break or torch.isnan(
            res_est).any()

    assert (err is not None), "Must provide err information when not in judgment mode"
    prefix, color = ('', 'red') if name == 'forward' else ('back_', 'blue')
    eval_prefix = '' if training else 'eval_'

    # Case 1: A nan entry is produced in Broyden
    if torch.isnan(res_est).any():
        msg = colored(f"WARNING: nan found in Broyden's {name} result. Diff: {diff}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}nan.pkl', 'wb'))
        return (1, msg, res_info)

    # Case 2: Unknown problem with Broyden's method (probably due to nan update(s) to the weights)
    if nstep == 0 and (diff != diff or diff > eps):
        msg = colored(f"WARNING: Bad Broyden's method {name}. Why?? Diff: {diff}. STOP.", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}badbroyden.pkl', 'wb'))
        return (2, msg, res_info)

    # Case 3: Protective break during Broyden (so that it does not diverge to infinity)
    if prot_break:
        msg = colored(f"WARNING: Hit Protective Break in {name}. Diff: {diff}. Total Iter: {len(trace)}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}prot_break.pkl', 'wb'))
        return (3, msg, res_info)

    return (-1, '', res_info)
