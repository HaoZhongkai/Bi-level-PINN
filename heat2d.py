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
parser.add_argument('--start_epochs', default=6000, type=int)
parser.add_argument("--epochs", default=500, type=int
                    )
parser.add_argument('--ft_steps', default=50, type=int
                    )
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--log_path',
                    default=None,
                    type=str)

parser.add_argument('--ift_method', default='rad', type=str)
# neumann sgd parameters
parser.add_argument('--neumann_iter', default=16, type=int)
parser.add_argument('--alpha', default=0.001, type=float)
# broyden parameters
parser.add_argument('--threshold', default=30, type=int)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--ls', default=True, type=bool)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--max_iter', default=-1, type=int)

# unrolled diff parameters
parser.add_argument('--rad_steps', default=5, type=int)

parser.add_argument('--gpu',
                    default=0,
                    type=int)

args = parser.parse_args()
print(args)
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Running on GPU ' + str(args.gpu))
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# if you want to use given CUDA, must set default device before import dde, if cpu, set after it, or set visible device to ''
import deepxde as dde
from deepxde.icbcs import DirichletBC, PeriodicBC, IC
from bpinn.pdeco import PDEOptimizerModel
from bpinn.utils import ParametricDirichletBC
from bpinn.utils import ResampleCallback

'''
    Control of 2d heat equation by the domain function
    ut - nu*(uxx + uyy) = f(t)

    u(x,0) = 0
    u(x,1) = 0
    u(0,y) = 0
    u(1,y) = 0

    Solution
    None

    Goal 
    J(u) = 1/2 *\int_0^2 |u - u_ref|^2 dx
   


    Initial 
    None


'''

pi = np.pi

nu = 0.001


# tensorized
def fun_ref(x):
    return 16 * pi * x[:, 0:1] * x[:, 1:2] * (x[:, 0:1] - 1) * (x[:, 1:2] - 1) * torch.cos(pi * x[:, 2:3]) - 32 * nu * (
            x[:, 0:1] * (x[:, 0:1] - 1) + x[:, 1:2] * (x[:, 1:2] - 1)) * torch.sin(pi * x[:, 2:3])


class Heat2d():
    def __init__(self, net_c):
        self.net_c = net_c

        self.f_ref = fun_ref

    def __call__(self, x, u):
        uxx = dde.grad.hessian(u, x, i=0, j=0)
        uyy = dde.grad.hessian(u, x, i=1, j=1)
        ut = dde.grad.jacobian(u, x, i=0, j=2)
        return ut - nu * (uxx + uyy) - self.net_c(x[:, 2:3])


class LaplaceControler(PDEOptimizerModel):
    def __init__(self, net_u, net_c, loss=None, log_path=None, n_domain=256, n_boundary=64, n_initial=64):
        pde = Heat2d(net_c)

        def boundary(x, on_boundary):
            return on_boundary and (
                    np.isclose(x[0], 0) or np.isclose(x[0], 1) or np.isclose(x[1], 0) or np.isclose(x[1], 1))

        def u_ref(x):
            return 16 * x[:, 0:1] * x[:, 1:2] * (1 - x[:, 0:1]) * (1 - x[:, 1:2]) * np.sin(pi * x[:, 2:3])

        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        timedomain = dde.geometry.TimeDomain(0, 2)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        bc = DirichletBC(geomtime, lambda _: 0, boundary)
        ic = IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)
        data = dde.data.TimePDE(geomtime, pde, [bc, ic], n_domain, n_boundary, n_initial, solution=u_ref, num_test=512)

        self.theta_loss_ref = self.theta_loss

        super(LaplaceControler, self).__init__(data, net_u, loss=None, theta=net_c, log_path=log_path)

    # calcalate loss for reference control parameters
    def theta_loss(self, theta):
        with torch.no_grad():
            x = torch.tensor(self.data.geom.uniform_points(32 * 32 * 20), requires_grad=True)
            y_ref = fun_ref(x)
            y = self.theta(x[:, 2:3])
            return ((y - y_ref) ** 2).mean()

    # sample uniform points space X time
    def evaluate_J(self, n):
        x = torch.tensor(self.data.geom.uniform_points(32 * 32 * 20), requires_grad=True)
        u = self.net(x)
        return ((u - 16 * x[:, 0:1] * x[:, 1:2] * (1 - x[:, 0:1]) * (1 - x[:, 1:2]) * torch.sin(
            pi * x[:, 2:3])) ** 2).mean()

    def save_source(self):
        N = 64
        x = torch.linspace(0, 2, N)[:, None]
        f = self.theta(x).detach().cpu().numpy()
        x_np = np.linspace(0, 2, N)[:, None]
        f_x = np.concatenate([x_np, f], axis=1)

        if self.log_path is not None:
            source_path = self.log_path + '_source.txt'
        else:
            source_path = './heat2d_source.txt'
        np.savetxt(source_path, f_x)


def main():
    log_dir = './data/models/' + str(args.log_path) + time.strftime(
        '_%m%d/%H_%M_%S') if args.log_path else None  # log folder by day
    if log_dir:
        sys.stdout = open('./data/logs/' + str(args.log_path) + time.strftime('_%m%d_%H_%M_%S') + '.txt', 'w')
    print(args)

    load_path = None
    # load_path = 'laplace_domain.pth'

    layer_size = [3] + [64] * 4 + [1]

    activation = "tanh"
    initializer = "Glorot normal"

    net_u = dde.maps.FNN(layer_size, activation, initializer)
    net_c = dde.maps.FNN([1, 32, 32, 1], activation, initializer)

    # net_u.apply_output_transform(hard_constraint)

    model = LaplaceControler(net_u, net_c, log_path=log_dir, n_domain=32 * 32, n_boundary=4 * 64, n_initial=4 * 64)

    # training initial configuration

    model.compile("adam", lr=0.001, metrics=['l2 relative error'])
    losshistory, train_state = model.train(epochs=args.start_epochs, display_every=1000)

    model.train_pdeco(epochs=args.epochs,
                      finetune_epochs=args.ft_steps,
                      num_val=64,
                      lr=args.lr,
                      ifd_method=args.ift_method,
                      threshold=args.threshold,
                      ls=args.ls,
                      eps=args.eps,
                      beta=args.beta,
                      neumann_iter=args.neumann_iter,
                      alpha=args.alpha,
                      rad_steps=args.rad_steps,
                      grad_ref=None
                      )


if __name__ == "__main__":
    main()
