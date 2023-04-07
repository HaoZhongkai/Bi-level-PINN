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
parser.add_argument('--start_epochs', default=5000, type=int)
parser.add_argument("--epochs", default=500, type=int
                    )
parser.add_argument('--ft_steps', default=50, type=int
                    )
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--log_path',
                    default=None,
                    type=str)

parser.add_argument('--ift_method', default='broyden', type=str)
# neumann sgd parameters
parser.add_argument('--neumann_iter', default=16, type=int)
parser.add_argument('--alpha', default=0.001, type=float)
# broyden parameters
parser.add_argument('--threshold', default=40, type=int)
parser.add_argument('--eps', default=1e-4, type=float)
parser.add_argument('--ls', default=True, type=bool)
parser.add_argument('--beta', default=0.01, type=float)
#
parser.add_argument("--rad_steps", default=5, type=int)
parser.add_argument('--gpu',
                    default=1,
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
from bpinn.dde_utils import ParametricDirichletBC, ParametricPointSetBC
from bpinn.dde_utils import ResampleCallback
import torch.nn.functional as F

pi = np.pi

width = 1.5
height = 1.0
v_inlet2_max = 0.5
Re = 100

'''
    Use FEM solver to estimate the result formally
'''


class NSControler(PDEOptimizerModel):
    def __init__(self, net_u, net_c, loss=None, log_path=None, n_domain=256, n_boundary=64, n_initial=64):

        def ns_pde(x, u):
            u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (
                    (u_vel * u_vel_x + v_vel * u_vel_y) + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
            )
            momentum_y = (
                    (u_vel * v_vel_x + v_vel * v_vel_y) + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
            )
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde

        rec = dde.geometry.Rectangle(
            xmin=[0, 0],
            xmax=[width, height]
        )
        geom = rec

        def boundary_left(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        def boundary_right(x, on_boundary):
            return on_boundary and np.isclose(x[0], width)

        def boundary_up_down(x, on_boundary):
            return on_boundary and ((np.isclose(x[1], 0) and (x[0] <= 1 / 3 * width or x[0] >= 2 / 3 * width)) or (
                    np.isclose(x[1], height) and (x[0] <= 1 / 3 * width or x[0] >= 2 / 3 * width)))

        def boundary_inlet2(x, on_boundary):
            return on_boundary and ((np.isclose(x[1], 0) and (1 / 3 * width < x[0] < 2 / 3 * width)) or (
                    np.isclose(x[1], height) and (1 / 3 * width < x[0] < 2 / 3 * width)))

        def u_inlet(x, theta):
            return 4 * x[:, 1:2] * (height - x[:, 1:2]) * (1)

        def u_target(x):
            return 4 * x[:, 1:2] * (height - x[:, 1: 2])

        def v_inlet2(x):
            return 36 * v_inlet2_max * (x[:, 0:1] - 1 / 3 * width) * (2 / 3 * width - x[:, 0:1]) / (width ** 2)
            # BCs

        def get_inlet_points(n):
            return np.concatenate([np.zeros([n])[:, None], np.linspace(0, height, n)[:, None]], axis=1)

        def aux_constraint(x, theta):
            return theta * x[:, 1:2] * (1 - x[:, 1:2]) * 4

        inlet_points = get_inlet_points(net_c.shape[0])
        bc_l_u = ParametricPointSetBC(inlet_points, net_c, component=0, aux_fun=aux_constraint)
        # bc_l_u = ParametricDirichletBC(geom, u_inlet, boundary_left, component=0,theta=net_c)
        bc_l_v = DirichletBC(geom, lambda _: 0, boundary_left, component=1)
        bc_r = DirichletBC(geom, lambda _: 0, boundary_right, component=2)

        bc_wall_u = DirichletBC(geom, lambda _: 0, boundary_up_down, component=0)
        bc_wall_v = DirichletBC(geom, lambda _: 0, boundary_up_down, component=1)

        bc_inlet2_u = DirichletBC(geom, lambda _: 0, boundary_inlet2, component=0)
        bc_inlet2_v = DirichletBC(geom, v_inlet2, boundary_inlet2, component=1)

        self.bcs = [bc_l_u, bc_l_v, bc_r, bc_wall_u, bc_wall_v, bc_inlet2_u, bc_inlet2_v]

        data = dde.data.PDE(geom, self.pde, self.bcs, n_domain, n_boundary, solution=None)

        self.theta_loss_ref = None
        self.u_target = u_target
        super(NSControler, self).__init__(data, net_u, loss=None, theta=net_c, log_path=log_path)

    # calcalate loss for reference control parameters
    def theta_loss(self, theta):
        pass

    # sample uniform points space X time
    def evaluate_J(self, n):
        x = torch.cat([width * torch.ones([n, 1]), torch.linspace(0, height, n).unsqueeze(-1)], dim=1)
        x.requires_grad = True
        u = self.net(x)
        J = height * torch.mean((u[:, 0:1] - self.u_target(x)) ** 2)

        ## penalize the source function
        z = torch.linspace(0, 1, len(self.theta))[:, None]
        u_in = self.theta * z * (1 - z) * 4
        reg = 100 * (F.relu(-self.theta) ** 2 + F.relu(self.theta - 4) ** 2).mean() + 100.0 * F.relu(
            self.theta.mean() - 1.5) ** 2

        return (J + reg)

    def save_source(self):
        if self.log_path is not None:
            source_path = self.log_path + 'ns2inlet_source.txt'
        else:
            source_path = './ns2inlet_source.txt'
        with torch.no_grad():
            x = torch.linspace(0, 1, len(self.theta))[:, None]
            source = self.theta.clone()
            x, source = x.cpu().numpy(), source.cpu().numpy()
            f_source = np.concatenate([x, source], axis=1)
            np.savetxt(source_path, f_source)
        return


def main():
    log_dir = './data/models/' + str(args.log_path) + time.strftime(
        '_%m%d/%H_%M_%S') if args.log_path else None  # log folder by day
    if log_dir:
        sys.stdout = open('./data/logs/' + str(args.log_path) + time.strftime('_%m%d_%H_%M_%S') + '.txt', 'w')
    print(args)

    load_path = None

    layer_size = [2] + [64] * 5 + [3]

    activation = "tanh"
    initializer = "Glorot uniform"

    net_u = dde.maps.FNN(layer_size, activation, initializer)
    net_c = (torch.ones([64]) + 0.0 * torch.randn([64])).unsqueeze(-1).requires_grad_(True)

    model = NSControler(net_u, net_c, log_path=log_dir, n_domain=64 * 64, n_boundary=4 * 64)
    lossweights = [1.0] * 3 + [1.0] + [1.0] * 6
    # training initial configuration
    if load_path is None:
        model.compile("adam", lr=0.001, metrics=None, loss_weights=lossweights)
        losshistory, train_state = model.train(epochs=args.start_epochs, display_every=2000)

    else:
        model.compile("adam", lr=0.001)
        net_u.load_state_dict(torch.load(load_path, map_location='cuda' if args.gpu else "cpu")['net_u'])
        net_c.load_state_dict(torch.load(load_path, map_location='cuda' if args.gpu else "cpu")['net_c'])
        model.train(epochs=1000, display_every=1000)

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
