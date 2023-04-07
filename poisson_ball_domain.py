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
parser.add_argument('--start_epochs', default=10000, type=int)
parser.add_argument("--epochs", default=20, type=int
                    )
parser.add_argument('--ft_steps', default=256, type=int
                    )
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--log_path',
                    default=None,
                    type=str)

parser.add_argument('--ift_method', default='broyden', type=str)
# neumann sgd parameters
parser.add_argument('--neumann_iter', default=16, type=int)
parser.add_argument('--alpha', default=0.001, type=float)
# broyden parameters
parser.add_argument('--threshold', default=20, type=int)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--ls', default=True, type=bool)
# trad parameters
parser.add_argument('--rad_steps', default=5, type=int)

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
from deepxde.icbcs import DirichletBC, PeriodicBC
from bpinn.pdeco import PDEOptimizerModel
from bpinn.dde_utils import ParametricDirichletBC

dde.config.set_default_float("float32")

'''
    Note that definition of objective function reported in this file is not the same with the paper, 
    a constant of the area is multiplied.
'''

width = 8
height = 8
ctrl_r = 1.6
pi = np.pi
Re = 100


class Possion2d():
    def __init__(self, net_c):
        self.net_c = net_c

    def __call__(self, x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        return u_xx + u_yy - self.net_c(x) * (torch.norm(x, dim=1, keepdim=True) <= ctrl_r)


class LaplaceControler(PDEOptimizerModel):
    def __init__(self, net_u, net_c, loss=None, log_path=None, n_domain=256, n_boundary=64):

        self.pde = Possion2d(net_c)

        x_bou_ctrl = 4 * np.array([
            [0.6, 0.6, 0.2],
            [0.6, -0.6, 0.2],
            [-0.6, 0.6, 0.2],
            [-0.6, -0.6, 0.2]
        ])
        rec = dde.geometry.Rectangle(
            xmin=[-width / 2, -height / 2],
            xmax=[width / 2, height / 2]
        )
        geom = rec
        disks = []
        for i in range(len(x_bou_ctrl)):
            disk = dde.geometry.Disk(
                x_bou_ctrl[i][0: 2],
                x_bou_ctrl[i][2]
            )

            disks.append(disk)
            geom = dde.geometry.csg.CSGDifference(geom, disk)

        self.geom = geom
        self.ctrl_area = dde.geometry.Disk([0, 0], 1.6)

        def boundary_rec(x, on_boundary):
            return on_boundary and (np.isclose(x[0], -width / 2) or np.isclose(x[0], width / 2) or np.isclose(x[1],
                                                                                                              -height / 2) or np.isclose(
                x[1], height / 2))

        def boundary_circle(x, on_boundary):
            return on_boundary and (not rec.on_boundary(x))

        # BCs

        bc_rec = DirichletBC(geom, lambda _: 1, boundary_rec, component=0)
        bc_circle = DirichletBC(geom, lambda _: 0, boundary_circle, component=0)

        self.bcs = [bc_rec, bc_circle]

        data = dde.data.PDE(geom, self.pde, [bc_rec, bc_circle], n_domain, n_boundary)

        self.theta_loss_ref = self.theta_loss

        super(LaplaceControler, self).__init__(data, net_u, loss=None, theta=net_c, log_path=log_path)

    # calcalate loss for reference control parameters
    def theta_loss(self, theta):
        pass

    def sample_eval_points(self, n):
        pass

    # sample top boundary points
    def evaluate_J(self, n):
        x = torch.tensor(self.data.geom.uniform_points(n), requires_grad=True)
        u = self.net(x)
        return ((u - 1) ** 2).mean() * (width * height - 4 * pi * 0.8 ** 2)

    def save_source(self):
        if self.log_path is not None:
            source_path = self.log_path + '_source.txt'
        else:
            source_path = './data/possion_ball.txt'
        N = 24 * 24
        x = torch.tensor(self.ctrl_area.random_points(N), requires_grad=True)
        f_source = self.theta(x)
        x, f_source = x.detach().cpu().numpy(), f_source.detach().cpu().numpy()
        f_x = np.concatenate([x, f_source], axis=1)

        np.savetxt(source_path, f_x)
        torch.save({'net_u': self.net.state_dict(), 'net_c': self.theta.state_dict()}, 'possion_ball.pth')

        return f_x


def main():
    log_dir = './data/models/' + str(args.log_path) + time.strftime(
        '_%m%d/%H_%M_%S') if args.log_path else None  # log folder by day
    if log_dir:
        sys.stdout = open('./data/logs/' + str(args.log_path) + time.strftime('_%m%d_%H_%M_%S') + '.txt', 'w')
        # sys.stdout = sys.__stdout__
    print(args)

    load_path = None
    # load_path = 'laplace_0.pth'

    layer_size = [2] + [50] * 4 + [1]
    activation = "tanh"
    initializer = "Glorot normal"

    net_u = dde.maps.FNN(layer_size, activation, initializer)
    net_c = dde.maps.FNN([2, 32, 32, 32, 1], activation, initializer)
    # net_c = torch.tensor([-0.1],requires_grad=True)
    # net_c = torch.tensor([1.0,1.0,1.0],requires_grad=True)

    model = LaplaceControler(net_u, net_c, log_path=log_dir, n_domain=48 * 48, n_boundary=8 * 48)

    # training initial configuration
    if load_path is None:
        model.compile("adam", lr=0.001)
        losshistory, train_state = model.train(epochs=args.start_epochs, display_every=2000)
        # model.compile("L-BFGS",metrics=['l2 relative error'])
        # losshistory, train_state = model.train(epochs=6000)
        # model.compile("adam", lr=0.001)
        # model.train(epochs=3000)
    else:
        model.compile("adam", lr=0.001)
        # model.restore(load_path,verbose=True)
        net_u.load_state_dict(torch.load(load_path, map_location='cuda' if args.gpu else "cpu")['net_u'])
        # net_c.load_state_dict(torch.load(load_path, map_location='cuda' if args.gpu else "cpu")['net_c'])
        # model.net.load_state_dict(torch.load(load_path,map_location='cuda' if args.gpu else 'cpu')['model_state_dict'])
        model.train(epochs=1000, display_every=1000)

    model.train_pdeco(epochs=args.epochs,
                      finetune_epochs=args.ft_steps,
                      num_val=48 * 48,
                      lr=args.lr,
                      ifd_method=args.ift_method,
                      threshold=args.threshold,
                      eps=args.eps,
                      neumann_iter=args.neumann_iter,
                      alpha=args.alpha,
                      rad_steps=args.rad_steps,
                      grad_ref=None
                      )

    torch.save({'net_u': net_u.state_dict(), 'net_c': net_c.state_dict()}, 'possion_ball.pth')


if __name__ == "__main__":
    main()
