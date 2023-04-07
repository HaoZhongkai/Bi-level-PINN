#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import sys

sys.path.append('..')

import numpy as np
import torch
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=50, type=int
                    )
parser.add_argument('--ft_steps', default=50, type=int
                    )
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--log_path',
                    default=None,
                    type=str)

parser.add_argument('--ift_method', default='cg', type=str)
# neumann sgd parameters
parser.add_argument('--neumann_iter', default=30, type=int)
parser.add_argument('--alpha', default=0.001, type=float)
# broyden parameters
parser.add_argument('--threshold', default=500, type=int)
parser.add_argument('--eps', default=1e-6, type=float)
# rad parameters
parser.add_argument('--rad_steps', default=5, type=int)

parser.add_argument('--ls', default=True, type=bool)

parser.add_argument('--gpu',
                    default=1,
                    type=int)

args = parser.parse_args()
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Running on GPU ' + str(args.gpu))

print(args)

# set device before import dde module
import deepxde as dde
from bpinn.pdeco import PDEOptimizerModel
from bpinn.dde_utils import ParametricDirichletBC

'''
    Control of Possion equation
    y'' = 2
    y(0)=theta_0, y(1)=theta_1

    
    Solution
    y = x**2+(theta_1-theta_0-1)x+theta_0
    
    Goal 
    y = x**2,  theta = (0, 1)
    
    Initial 
    theta = (1,1)
    
    J = 1/3*(theta_0**2 + theta_1**2 + theta_1*theta_0 -2*theta_1-theta_0 +1)
    
    J_grad = 1/3*(2*theta_0+theta_1 - 1, 2*theta_1+theta_0 -2)
    
    
'''


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2


def eva_J(x, y):
    return ((y - x ** 2) ** 2).mean()


def J_grad_ref(theta):
    theta_grad = torch.zeros_like(theta)
    theta_grad[0] = (2 * theta[0] + theta[1] - 1) / 3
    theta_grad[1] = (2 * theta[1] + theta[0] - 2) / 3
    return theta_grad


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


# tensorized function
def left_u(x, theta):
    return theta[0]


def right_u(x, theta):
    return theta[1]


def main():
    log_dir = './data/models/' + str(args.log_path) + time.strftime(
        '_%m%d/%H_%M_%S') if args.log_path else None  # log folder by day
    if log_dir:
        sys.stdout = open('./data/logs/' + str(args.log_path) + time.strftime('_%m%d_%H_%M_%S') + '.txt', 'w')
    print(args)

    theta = torch.tensor([1.0, 1.0], requires_grad=True)
    theta_ref = torch.tensor([0.0, 1.0], requires_grad=False)
    geom = dde.geometry.Interval(0, 1)
    # bc_l = DirichletBC(geom, lambda _: 1, boundary_l)
    # bc_r = DirichletBC(geom, lambda _: 0, boundary_r)
    bc_l = ParametricDirichletBC(geom, left_u, boundary_l, theta=theta)  # seems a little slower
    bc_r = ParametricDirichletBC(geom, right_u, boundary_r, theta=theta)
    data = dde.data.PDE(geom, pde, [bc_l, bc_r], 64, 16, num_test=100)

    layer_size = [1] + [50] * 4 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    # model = dde.Model(data, net)
    model = PDEOptimizerModel(data, net, eva_J, theta, log_path=log_dir)
    model.compile("adam", lr=0.001)

    losshistory, train_state = model.train(epochs=1000, display_every=500)
    # losshistory, train_state = None, None

    model.train_pdeco(epochs=args.epochs,
                      finetune_epochs=args.ft_steps,
                      num_val=64,
                      lr=args.lr,
                      ifd_method=args.ift_method,
                      threshold=args.threshold,
                      eps=args.eps,
                      neumann_iter=args.neumann_iter,
                      alpha=args.alpha,
                      rad_steps=args.rad_steps,
                      grad_ref=J_grad_ref,
                      theta_loss_ref=theta_ref,
                      )

    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    dde.postprocessing.plot_loss_history(loss_history=losshistory, fname=log_dir + '/loss.png' if log_dir else None)

    # model.save('possion',protocol='backend')


if __name__ == "__main__":
    main()
