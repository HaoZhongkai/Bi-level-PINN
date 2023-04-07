import os
import sys

import torch
import time
import numpy as np
import copy
from torch.autograd import grad
import deepxde as dde
from deepxde.model import Model
from deepxde.geometry.geometry import Geometry
from torch.utils.tensorboard import SummaryWriter


class ParamDomain(Geometry):
    def __init__(self, geom, param_sampler, param_dim, mixed_training=True, n_param_batch=1):
        super(ParamDomain, self).__init__(geom.dim, geom.bbox, geom.diam)
        self.geom = geom
        self.param_sampler = param_sampler  # generate N*n
        self.n_param_batch = n_param_batch
        self.mixed_training = mixed_training  # train one batch with random boundary conditions
        self.dim = self.geom.dim + param_dim

        self.param_batch = None

    def inside(self, x):
        return self.geom.inside(x[:, 0:self.geom.dim])

    def on_boundary(self, x):
        return self.geom.on_boundary(x[:, 0:self.geom.dim])

    def uniform_points(self, n, boundary=True):
        if boundary:
            return np.concatenate([self.random_points(n), self.random_boundary_points(n)], axis=0)
        else:
            return self.random_points(n)

    # total number of points: n , per param n // n_param_batch
    def random_points(self, n, random="pseudo"):
        m = (n // self.n_param_batch)
        x = np.empty(shape=(m * self.n_param_batch, self.dim))
        if self.mixed_training:
            self.param_batch = self.param_sampler([self.n_param_batch, m])
        else:
            self.param_batch = self.param_sampler([self.n_param_batch, 1])
        for i in range(self.n_param_batch):
            x[i * m: (i + 1) * m, :self.geom.dim] = self.geom.random_points(n, random=random)
            x[i * m: (i + 1) * m, self.geom.dim:] = self.param_batch[i][..., np.newaxis]
        return x

    def random_boundary_points(self, n, random="pseudo"):
        m = (n // self.n_param_batch)
        x = np.empty(shape=((n // self.n_param_batch) * self.n_param_batch, self.dim))
        if self.mixed_training:
            self.param_batch = self.param_sampler([self.n_param_batch, m])
        elif self.param_batch is None:
            self.param_batch = self.param_sampler([self.n_param_batch, 1])
        for i in range(self.n_param_batch):
            x[i * m: (i + 1) * m, :self.geom.dim] = self.geom.random_boundary_points(n, random=random)
            x[i * m: (i + 1) * m, self.geom.dim:] = self.param_batch[i][..., np.newaxis]
        return x


import deepxde.utils as utils
import deepxde.backend as bkd
from bpinn.utils import ParametricDirichletBC, ParametricPointSetBC
from deepxde.callbacks import CallbackList
from bpinn.ifd_grad import paddedlist, implicit_gradient_neumann_net, implicit_gradient_neumann_net2, \
    implicit_gradient_broyden_net, implicit_gradient_broyden_net2, implicit_adjoint_hessian_product_broyden, \
    implicit_adjoint_hessian_product_neumann, implicit_gradient_t1t2_net, implicit_gradient_t1t2_net2
from torch.autograd import grad


class PDEOptimizerModel(dde.Model):
    def __init__(self, data, net, loss=None, theta=None, log_path=None):
        super(PDEOptimizerModel, self).__init__(data, net)

        self.obj_loss = loss
        self.theta = theta
        self.log_path = log_path
        self.writer = SummaryWriter(log_dir=log_path, comment='obj value') if log_path else None

    def sample_test_interior_points(self, n):
        return self.data.geom.uniform_points(n, boundary=False)

    def sample_test_boundary_points(self, n):
        raise NotImplementedError

    # default loss, L2 loss compared with a reference function
    def evaluate_J(self, n_domain):
        x = torch.tensor(self.sample_test_interior_points(n_domain), requires_grad=True)
        u = self.net(x)
        J = self.obj_loss(x, u)

        return J

    # TODO: Now only registered to parametric Dirichlet boundary
    def update_boundary_condition(self):
        for bc in self.data.bcs:
            if isinstance(bc, (ParametricDirichletBC, ParametricPointSetBC)):
                bc.theta = self.theta
        return

    def save_source(self, *args):
        pass

    '''
    
        Main training function for PDE constrained optimization
        
    '''

    @utils.timing
    def train_pdeco(self,
                    epochs=2,
                    lr=0.05,
                    finetune_epochs=100,
                    num_val=1000,
                    ifd_method='neumann',
                    neumann_iter=30,
                    alpha=0.001,
                    threshold=30,
                    max_iter=-1,
                    eps=1e-6,
                    ls=True,
                    beta=1e-3,
                    rad_steps=5,
                    grad_ref=None,
                    theta_loss_ref=None,
                    callbacks=None,
                    model_save_path=None,
                    ):
        self.callbacks = CallbackList(callbacks=callbacks)
        self.optimizer = torch.optim.Adam(
            [self.theta] if isinstance(self.theta, torch.Tensor) else self.theta.parameters(), lr=lr, capturable=True)
        # self.optimizer = torch.optim.SGD([self.theta] if isinstance(self.theta, torch.Tensor) else self.theta.parameters(),lr=lr)

        self.lr = lr
        self.finetune_epochs = finetune_epochs
        self.num_val = num_val
        self.ifd_method = ifd_method
        self.neumann_iter = neumann_iter
        self.alpha = alpha
        self.threshold = threshold
        self.eps = eps
        self.ls = ls
        self.beta = beta
        self.rad_steps = rad_steps

        if not hasattr(self, 'theta_loss_ref'):
            self.theta_loss_ref = theta_loss_ref
        # self.optimizer = torch.optim.SGD([self.theta],lr=lr)

        for epoch in range(epochs):

            self._train_sgd(epochs=finetune_epochs, display_every=finetune_epochs // 4 if finetune_epochs >= 4 else 1)
            # print('Test PDE loss',np.sum(self.train_state.loss_test))

            J = self.evaluate_J(num_val)
            pde_loss = \
                self.outputs_losses(training=True, inputs=self.train_state.X_train, targets=self.train_state.y_train)[
                    1].sum()

            # update control parameters
            # two choices: 1. update boundary condition manually; 2. use inplace operation with torch.no_grad
            # with torch.no_grad():

            # optimizer update
            if isinstance(self.theta, torch.Tensor):

                if ifd_method == 'neumann':
                    J_grad = implicit_gradient_neumann_net(self.theta, list(self.net.parameters()), J, pde_loss,
                                                           n=neumann_iter, alpha=alpha)
                elif ifd_method == "broyden":
                    J_grad = implicit_gradient_broyden_net(self.theta, list(self.net.parameters()), J, pde_loss,
                                                           threshold=threshold, max_iter=max_iter, eps=eps, ls=ls,
                                                           beta=beta)
                elif ifd_method == 't1t2':
                    J_grad = implicit_gradient_t1t2_net(self.theta, list(self.net.parameters()), J, pde_loss)
                elif ifd_method == "rad":
                    J_grad = self.hypergradients_rad_nets()
                else:
                    raise NotImplementedError

                self.optimizer.zero_grad()
                self.theta.grad = J_grad
            else:

                if ifd_method == 'neumann':
                    J_grad = implicit_gradient_neumann_net2(list(self.theta.parameters()), list(self.net.parameters()),
                                                            J, pde_loss, n=neumann_iter, alpha=alpha)
                elif ifd_method == "broyden":
                    J_grad = implicit_gradient_broyden_net2(list(self.theta.parameters()), list(self.net.parameters()),
                                                            J, pde_loss, threshold=threshold, max_iter=max_iter,
                                                            eps=eps, ls=ls, beta=beta)
                elif ifd_method == 't1t2':
                    J_grad = implicit_gradient_t1t2_net2(list(self.theta.parameters()), list(self.net.parameters()), J,
                                                         pde_loss)
                elif ifd_method == "rad":
                    J_grad = self.hypergradients_rad_nets()
                else:
                    raise NotImplementedError

                self.optimizer.zero_grad()
                for i, p in enumerate(self.theta.parameters()):
                    p.grad = J_grad[i].clip(-0.1, 0.1)  # whether to try gradient clipping
            self.optimizer.step()
            self.update_boundary_condition()

            theta_loss, grad_sim = None, None
            self.epoch = epoch
            if self.theta_loss_ref is not None:  # function or tensor
                if isinstance(self.theta_loss_ref, torch.Tensor):
                    theta_loss = torch.sum((self.theta - self.theta_loss_ref) ** 2)
                else:
                    theta_loss = self.theta_loss_ref(self.theta)  # might be a tensor or net
            if self.writer:
                self.writer.add_scalar('PDE loss', pde_loss, epoch)
                self.writer.add_scalar('J value', J, epoch)
                if grad_ref:
                    grad_sim = (J_grad * grad_ref(self.theta)).sum() / (
                            torch.norm(J_grad, p=2) * torch.norm(grad_ref(self.theta), p=2))

                    self.writer.add_scalar('Grad Similarity', grad_sim, epoch)

                if theta_loss is not None:
                    self.writer.add_scalar('Opt param loss', theta_loss, epoch)

            self.save_source()

            print('Epoch {} theta loss {} PDE loss {} J value {}'.format(epoch,
                                                                         theta_loss if theta_loss is not None else None,
                                                                         np.sum(self.train_state.loss_test), J.item()))

    def hypergradients_rad_nets(self):
        lr = 0.001
        nets = [self.net] + [copy.deepcopy(self.net) for _ in range(self.rad_steps)]
        params_grad_history = []
        for i in range(self.rad_steps):
            E = self.outputs_losses(training=True, inputs=self.train_state.X_train, targets=self.train_state.y_train)[
                1].sum()
            param = list(nets[i].parameters())
            param_grad = grad(E, param, create_graph=True)
            for p_net_new, p_net, p_g in zip(nets[i + 1].parameters(), nets[i].parameters(), param_grad):
                p_net_new.data = p_net.data - lr * p_g
            params_grad_history.append(param_grad)
            self.net = nets[i + 1]

        J = self.evaluate_J(self.num_val)
        if isinstance(self.theta, torch.Tensor):
            Jw_theta = torch.zeros_like(self.theta)
            v = grad(J, self.net.parameters(), retain_graph=True)
            for i in range(self.rad_steps - 1, -1, -1):
                gw_theta = \
                    grad(params_grad_history[i], self.theta, grad_outputs=v, retain_graph=True, allow_unused=True)[0]
                if gw_theta is None:
                    gw_theta = torch.zeros_like(self.theta)
                Jw_theta -= lr * gw_theta
                param_i = nets[i].parameters()
                v_temp = paddedlist(grad(params_grad_history[i], param_i, grad_outputs=v, retain_graph=True), param_i)
                for i, v_ in enumerate(v):
                    v_ -= lr * v_temp[i]
            J_theta = grad(J, self.theta, retain_graph=True, allow_unused=True)[0]
            if J_theta is None:
                J_theta = torch.zeros_like(self.theta)
            J_grad = J_theta + Jw_theta
        else:
            Jw_theta = [torch.zeros_like(t) for t in self.theta.parameters()]
            v = grad(J, self.net.parameters(), retain_graph=True)
            for i in range(self.rad_steps - 1, -1, -1):
                gw_theta = paddedlist(
                    grad(params_grad_history[i], self.theta.parameters(), grad_outputs=v, retain_graph=True,
                         allow_unused=True), Jw_theta)
                for j, g in zip(Jw_theta, gw_theta):
                    j -= lr * g
                param_i = nets[i].parameters()
                v_temp = paddedlist(grad(params_grad_history[i], param_i, grad_outputs=v, retain_graph=True), param_i)
                for i, v_ in enumerate(v):
                    v_ -= lr * v_temp[i]
            J_theta = paddedlist(grad(J, self.theta.parameters(), retain_graph=True, allow_unused=True), Jw_theta)
            J_grad = [t + wt for (t, wt) in zip(J_theta, Jw_theta)]

        return J_grad


class PDEOptimizerModelWithLossMonitor(dde.Model):
    def __init__(self, data, net, loss=None, theta=None, log_path=None):
        super(PDEOptimizerModelWithLossMonitor, self).__init__(data, net)

        self.obj_loss = loss
        self.theta = theta
        self.log_path = log_path
        self.writer = SummaryWriter(log_dir=log_path, comment='obj value') if log_path else None

    def sample_test_interior_points(self, n):
        return self.data.geom.uniform_points(n, boundary=False)

    def sample_test_boundary_points(self, n):
        raise NotImplementedError

    # default loss, L2 loss compared with a reference function
    def evaluate_J(self, n_domain):
        raise NotImplementedError

    # TODO: Now only registered to parametric Dirichlet boundary
    def update_boundary_condition(self):
        for bc in self.data.bcs:
            if isinstance(bc, (ParametricDirichletBC, ParametricPointSetBC)):
                bc.theta = self.theta
        return

    def save_source(self):
        pass

    '''

        Main training function for PDE constrained optimization

    '''

    @utils.timing
    def train_pdeco(self,
                    epochs=2,
                    lr=0.05,
                    finetune_epochs=100,
                    finetune_epochs_max=1000,
                    pde_loss_threshold=1e-2,
                    num_val=1000,
                    ifd_method='neumann',
                    neumann_iter=30,
                    alpha=0.001,
                    threshold=30,
                    eps=1e-6,
                    ls=True,
                    beta=1e-3,
                    grad_ref=None,
                    theta_loss_ref=None,
                    callbacks=None,
                    model_save_path=None,
                    ):
        self.callbacks = CallbackList(callbacks=callbacks)
        self.optimizer = torch.optim.Adam(
            [self.theta] if isinstance(self.theta, torch.Tensor) else self.theta.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD([self.theta] if isinstance(self.theta, torch.Tensor) else self.theta.parameters(), lr=lr)

        if not hasattr(self, 'theta_loss_ref'):
            self.theta_loss_ref = theta_loss_ref
        # self.optimizer = torch.optim.SGD([self.theta],lr=lr)
        for epoch in range(epochs):

            for k in range(finetune_epochs_max // finetune_epochs):
                pde_loss = \
                    self.outputs_losses(training=True, inputs=self.train_state.X_train,
                                        targets=self.train_state.y_train)[
                        1].sum()
                if pde_loss < pde_loss_threshold:
                    print('pde loss {}'.format(pde_loss))
                    break
                self._train_sgd(epochs=finetune_epochs, display_every=finetune_epochs)

            # print('Test PDE loss',np.sum(self.train_state.loss_test))

            J = self.evaluate_J(num_val)
            pde_loss = \
                self.outputs_losses(training=True, inputs=self.train_state.X_train, targets=self.train_state.y_train)[
                    1].sum()

            # update control parameters
            # two choices: 1. update boundary condition manually; 2. use inplace operation with torch.no_grad
            # with torch.no_grad():

            # manually update
            # self.theta = self.theta + 0.03*torch.randn_like(self.theta) # test code
            # self.theta = self.theta - lr*J_grad/(torch.norm(J_grad))
            # optimizer update
            if isinstance(self.theta, torch.Tensor):

                if ifd_method == 'neumann':
                    J_grad = implicit_gradient_neumann_net(self.theta, list(self.net.parameters()), J, pde_loss,
                                                           n=neumann_iter,
                                                           alpha=alpha)
                elif ifd_method == "broyden":
                    J_grad = implicit_gradient_broyden_net(self.theta, list(self.net.parameters()), J, pde_loss,
                                                           threshold=threshold, eps=eps, ls=ls, beta=beta)
                else:
                    raise NotImplementedError

                self.optimizer.zero_grad()
                self.theta.grad = J_grad
            else:

                if ifd_method == 'neumann':
                    J_grad = implicit_gradient_neumann_net2(list(self.theta.parameters()), list(self.net.parameters()),
                                                            J,
                                                            pde_loss, n=neumann_iter, alpha=alpha)
                elif ifd_method == "broyden":
                    J_grad = implicit_gradient_broyden_net2(list(self.theta.parameters()), list(self.net.parameters()),
                                                            J, pde_loss, threshold=threshold, eps=eps, ls=ls, beta=beta)
                else:
                    raise NotImplementedError

                self.optimizer.zero_grad()
                for i, p in enumerate(self.theta.parameters()):
                    p.grad = J_grad[i].clip(-0.1, 0.1)  # whether to try gradient clipping
            self.optimizer.step()
            self.update_boundary_condition()

            theta_loss, grad_sim = None, None
            if self.theta_loss_ref is not None:  # function or tensor
                if isinstance(self.theta_loss_ref, torch.Tensor):
                    theta_loss = torch.sum((self.theta - self.theta_loss_ref) ** 2)
                else:
                    theta_loss = self.theta_loss_ref(self.theta)  # might be a tensor or net
            if self.writer:
                self.writer.add_scalar('PDE loss', pde_loss, epoch)
                self.writer.add_scalar('J value', J, epoch)
                if grad_ref:
                    grad_sim = (J_grad * grad_ref(self.theta)).sum() / (
                            torch.norm(J_grad, p=2) * torch.norm(grad_ref(self.theta), p=2))
                    self.writer.add_scalar('Grad Similarity', grad_sim, epoch)

                if theta_loss is not None:
                    self.writer.add_scalar('Opt param loss', theta_loss, epoch)

            self.save_source()

            print('Epoch {} theta loss {} PDE loss {} J value {}'.format(epoch,
                                                                         theta_loss if theta_loss is not None else None,
                                                                         np.sum(self.train_state.loss_test), J.item()))


class PenaltyPDEModel(dde.Model):
    def __init__(self, data, net, loss=None, theta=None, log_path=None, metrics=None):
        super(PenaltyPDEModel, self).__init__(data, net)

        self.obj_loss = loss
        self.theta = theta

        self.metrics = metrics

        self.writer = SummaryWriter(log_dir=log_path, comment='obj value') if log_path else None

    @utils.timing
    def train_pdeco(self,
                    n_pdes,
                    finetune_epochs=100,
                    mu_max=100,
                    mu0=0.1,
                    beta=2,
                    grad_ref=None,
                    theta_loss_ref=None,
                    callbacks=None,
                    model_save_path=None,
                    ):

        self.callbacks = CallbackList(callbacks=callbacks)

        if not hasattr(self, 'theta_loss_ref'):
            self.theta_loss_ref = theta_loss_ref
        epoch = 0
        self.epoch = 0
        self.finetune_epochs = finetune_epochs
        mu = mu0

        while mu < mu_max:

            print("-" * 80)
            print("Iteration {}: mu = {}".format(self.epoch, mu))

            loss_weights = [1 / n_pdes * mu] * n_pdes + [1]
            self.compile("adam", lr=0.001, loss_weights=loss_weights, metrics=self.metrics)
            losshistory, train_state = self.train(epochs=finetune_epochs,
                                                  display_every=finetune_epochs // 4 if finetune_epochs >= 4 else 1)
            # self._train_sgd(epochs=finetune_epochs, display_every=finetune_epochs//4 if finetune_epochs>=4 else 1)
            # losshistory = self.losshistory

            mu *= beta

            self.save_source()
            pde_loss = (losshistory.loss_test[-1][:n_pdes] / losshistory.loss_weights[:n_pdes]).sum()
            J = losshistory.loss_test[-1][-1]
            theta_loss, grad_sim = None, None

            if self.theta_loss_ref is not None:  # function or tensor
                if isinstance(self.theta_loss_ref, torch.Tensor):
                    theta_loss = torch.sum((self.theta - self.theta_loss_ref) ** 2)
                else:
                    theta_loss = self.theta_loss_ref(self.theta)  # might be a tensor or net
            if self.writer:
                self.writer.add_scalar('PDE loss', pde_loss, epoch)
                self.writer.add_scalar('J value', J, epoch)
                # if grad_ref:
                #     grad_sim = (J_grad * grad_ref(self.theta)).sum() / (
                #             torch.norm(J_grad, p=2) * torch.norm(grad_ref(self.theta), p=2))
                #     self.writer.add_scalar('Grad Similarity', grad_sim, epoch)

                if theta_loss is not None:
                    self.writer.add_scalar('Opt param loss', theta_loss, epoch)
            epoch += 1
            self.epoch = epoch

            print('Epoch {} theta loss {} PDE loss {} J value {}'.format(epoch,
                                                                         theta_loss if theta_loss is not None else None,
                                                                         pde_loss, J.item()))

    def save_source(self):
        pass


class AugmentLagrangianPDEModel(dde.Model):
    def __init__(self, data, net, loss=None, theta=None, log_path=None, metrics=None):
        super(AugmentLagrangianPDEModel, self).__init__(data, net)

        self.obj_loss = loss
        self.theta = theta

        self.metrics = metrics

        self.writer = SummaryWriter(log_dir=log_path, comment='obj value') if log_path else None

    @utils.timing
    def train_pdeco(self,
                    n_pdes,
                    n_boundary,
                    finetune_epochs=100,
                    mu_max=100,
                    mu0=0.1,
                    beta=2,
                    grad_ref=None,
                    theta_loss_ref=None,
                    callbacks=None,
                    model_save_path=None,
                    ):

        self.callbacks = CallbackList(callbacks=callbacks)

        if not hasattr(self, 'theta_loss_ref'):
            self.theta_loss_ref = theta_loss_ref
        epoch = 0
        self.epoch = 0
        mu = mu0

        x = self.data.train_x[np.sum(self.data.num_bcs):]
        lbds = [torch.zeros([len(x), 1]) for _ in range(n_pdes)]
        while mu < mu_max:

            print("-" * 80)
            print("Iteration {}: mu = {}".format(epoch, mu))

            residuals = self.predict(x, operator=self.data.pde)[:n_pdes]
            lbds = [lbd + 2 / 3 * mu * torch.tensor(r) for r, lbd in zip(residuals, lbds)]

            losses_al = [lambda _, y: torch.mean(lbd * y) for lbd in lbds]
            losses = ['MSE'] * n_pdes + losses_al + ['MSE'] * n_boundary + ['MSE']
            loss_weights = [1 / n_pdes * mu] * n_pdes + [1] * n_pdes + [1 / n_boundary * mu] * n_boundary + [1]
            self.compile("adam", lr=0.001, loss=losses, loss_weights=loss_weights, metrics=self.metrics)
            losshistory, train_state = self.train(epochs=finetune_epochs,
                                                  display_every=finetune_epochs // 4 if finetune_epochs >= 4 else 1)
            # self._train_sgd(epochs=finetune_epochs, display_every=finetune_epochs//4 if finetune_epochs>=4 else 1)
            # losshistory = self.losshistory

            mu *= beta

            self.save_source()
            epoch += 1
            self.epoch = epoch

            pde_loss = (losshistory.loss_test[-1][:n_pdes] / losshistory.loss_weights[:n_pdes]).sum()
            J = losshistory.loss_test[-1][-1]
            theta_loss, grad_sim = None, None

            if self.theta_loss_ref is not None:  # function or tensor
                if isinstance(self.theta_loss_ref, torch.Tensor):
                    theta_loss = torch.sum((self.theta - self.theta_loss_ref) ** 2)
                else:
                    theta_loss = self.theta_loss_ref(self.theta)  # might be a tensor or net
            if self.writer:
                self.writer.add_scalar('PDE loss', pde_loss, epoch)
                self.writer.add_scalar('J value', J, epoch)
                # if grad_ref:
                #     grad_sim = (J_grad * grad_ref(self.theta)).sum() / (
                #             torch.norm(J_grad, p=2) * torch.norm(grad_ref(self.theta), p=2))
                #     self.writer.add_scalar('Grad Similarity', grad_sim, epoch)

                if theta_loss is not None:
                    self.writer.add_scalar('Opt param loss', theta_loss, epoch)

            print('Epoch {} theta loss {} PDE loss {} J value {}'.format(epoch,
                                                                         theta_loss if theta_loss is not None else None,
                                                                         pde_loss, J.item()))

    def save_source(self):
        pass


class PenaltyLineSearchPDEModel(dde.Model):
    def __init__(self, data, net, loss=None, theta=None, log_path=None, metrics=None):
        super(PenaltyLineSearchPDEModel, self).__init__(data, net)

        self.obj_loss = loss
        self.theta = theta

        self.metrics = metrics

        self.writer = SummaryWriter(log_dir=log_path, comment='obj value') if log_path else None

    def save_ctrl(self):
        pass

    def save_source(self):
        pass

    @utils.timing
    def train_pdeco(self,
                    n_pdes,
                    finetune_epochs=100,
                    finetune_epochs_max=10000,
                    loss_threshold=1e-3,
                    mu_max=100,
                    mu0=0.1,
                    beta=2,
                    grad_ref=None,
                    theta_loss_ref=None,
                    callbacks=None,
                    model_save_path=None,
                    ):

        self.callbacks = CallbackList(callbacks=callbacks)

        if not hasattr(self, 'theta_loss_ref'):
            self.theta_loss_ref = theta_loss_ref
        epoch = 0
        self.epoch = 0
        ft_iter = 0
        mu = mu0

        while mu < mu_max:

            print("-" * 80)
            print("Iteration {}: mu = {}".format(epoch, mu))

            loss_weights = [1] * n_pdes + [mu]
            self.compile("adam", lr=0.001, loss_weights=loss_weights, metrics=self.metrics)
            losshistory, train_state = self.train(epochs=finetune_epochs,
                                                  display_every=finetune_epochs // 4 if finetune_epochs >= 4 else 1)
            # self._train_sgd(epochs=finetune_epochs, display_every=finetune_epochs//4 if finetune_epochs>=4 else 1)
            # losshistory = self.losshistory

            mu *= beta

            pde_loss = (losshistory.loss_test[-1][:n_pdes] / losshistory.loss_weights[:n_pdes]).sum()

            if pde_loss < loss_threshold:
                self.save_ctrl()
                ft_iter = 0
            else:
                ft_iter += 1
                if ft_iter <= (finetune_epochs_max // finetune_epochs):
                    print('PDE loss reaches the threshold, continue finetuning')
                else:
                    print('PDE loss reaches the threshold, continue finetuning')
                    break

            self.save_source()
            epoch += 1
            self.epoch = epoch

            J = losshistory.loss_test[-1][-1]
            theta_loss, grad_sim = None, None

            if self.theta_loss_ref is not None:  # function or tensor
                if isinstance(self.theta_loss_ref, torch.Tensor):
                    theta_loss = torch.sum((self.theta - self.theta_loss_ref) ** 2)
                else:
                    theta_loss = self.theta_loss_ref(self.theta)  # might be a tensor or net
            if self.writer:
                self.writer.add_scalar('PDE loss', pde_loss, epoch)
                self.writer.add_scalar('J value', J, epoch)
                # if grad_ref:
                #     grad_sim = (J_grad * grad_ref(self.theta)).sum() / (
                #             torch.norm(J_grad, p=2) * torch.norm(grad_ref(self.theta), p=2))
                #     self.writer.add_scalar('Grad Similarity', grad_sim, epoch)

                if theta_loss is not None:
                    self.writer.add_scalar('Opt param loss', theta_loss, epoch)

            print('Epoch {} theta loss {} PDE loss {} J value {}'.format(epoch,
                                                                         theta_loss if theta_loss is not None else None,
                                                                         pde_loss, J.item()))
