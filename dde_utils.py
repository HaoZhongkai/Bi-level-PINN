#!/usr/bin/env python  
# -*- coding:utf-8 _*-

import numpy as np
import torch
import deepxde as dde

from deepxde.callbacks import Callback

'''

    More utils class for DeepXDE, support for torch
    
'''

'''

    Callback

'''


class ResampleCallback(Callback):
    """Resample the training points for PDE losses every given period."""

    def __init__(self, period=100, resample_test=True, resample_train_bc=False):
        super(ResampleCallback, self).__init__()
        self.period = period
        self.resample_test = resample_test
        self.resample_train_bc = resample_train_bc
        self.num_bcs_initial = None
        self.epochs_since_last_resample = 0

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.period is None or self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0

        # resample test points, too
        if self.resample_test:
            self.model.data.test_x, self.model.data.test_y, self.model.data.test_aux_vars = None, None, None
            self.model.train_state.set_data_test(*self.model.data.test())
        # self.model.data.train_x, self.model.data.train_y, self.model.data.train_aux_vars = None, None, None
        # TODO: Check whether to reset the data.train_x_bc
        if self.resample_train_bc:
            self.model.data.train_x_bc = None
        # self.model.data.train_next_batch()
        self.model.data.resample_train_points()

        if not np.array_equal(self.num_bcs_initial, self.model.data.num_bcs):
            print("Initial value of self.num_bcs:", self.num_bcs_initial)
            print("self.model.data.num_bcs:", self.model.data.num_bcs)
            raise ValueError(
                "`num_bcs` changed! Please update the loss function by `model.compile`."
            )

    def on_train_end(self):
        self.epochs_since_last_resample = 0

        # resample test points, too
        if self.resample_test:
            self.model.data.test_x, self.model.data.test_y, self.model.data.test_aux_vars = None, None, None
            self.model.train_state.set_data_test(*self.model.data.test())
        # self.model.data.train_x, self.model.data.train_y, self.model.data.train_aux_vars = None, None, None
        # TODO: Check whether to reset the data.train_x_bc
        if self.resample_train_bc:
            self.model.data.train_x_bc = None
        # self.model.data.train_next_batch()
        self.model.data.resample_train_points()

        # TODO: check whether this is necessary
        # if not np.array_equal(self.num_bcs_initial, self.model.data.num_bcs):
        #     print("Initial value of self.num_bcs:", self.num_bcs_initial)
        #     print("self.model.data.num_bcs:", self.model.data.num_bcs)
        #     raise ValueError(
        #         "`num_bcs` changed! Please update the loss function by `model.compile`."
        #     )


'''

    Geometry

'''
from deepxde.geometry.geometry import Geometry
from deepxde.geometry.sampler import sample


class Ellipse(Geometry):
    def __init__(self, center, axis_radius):
        self.center = np.array(center)
        self.axis_radius = np.array(axis_radius)

        super(Ellipse, self).__init__(2, (self.center - self.axis_radius, self.center + self.axis_radius),
                                      np.max(self.axis_radius))

    def inside(self, x):
        return np.linalg.norm((x - self.center) / self.axis_radius, axis=-1) <= 1

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm((x - self.center) / self.axis_radius, axis=-1), 1)

    def distance2boundary_unitdirn(self, x, dirn):
        raise NotImplementedError

    def distance2boundary(self, x, dirn):
        raise NotImplementedError

    def mindist2boundary(self, x):
        raise NotImplementedError

    def boundary_normal(self, x):
        _n = np.array([self.axis_radius[1] / self.axis_radius[0], self.axis_radius[0] / self.axis_radius[1]]) * (
                x - self.center)
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * (self.on_boundary(x)[:, None])
        return _n

    def boundary_curvature(self, x):
        _n = ((np.array([self.axis_radius[1] / self.axis_radius[0], self.axis_radius[0] / self.axis_radius[1]]) * (
                x - self.center)) ** 2).sum(axis=-1, keepdims=True)
        curl = self.axis_radius[0] * self.axis_radius[1] / ((_n) ** (1.5))
        return curl * (self.on_boundary(x)[:, None])

    def random_points(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.axis_radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        raise NotImplementedError

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
        return self.axis_radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError


'''

    Boundary

'''
import deepxde.backend as bkd
import numbers
from deepxde import config
from deepxde.icbcs import BC
# from deepxde.icbcs.boundary_conditions import npfunc_range_autocache
from functools import wraps


class ParametricDirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0, theta=None):
        super(ParametricDirichletBC, self).__init__(geom, on_boundary, component)
        self.func = parametric_npfunc_range_autocache(func)
        self.theta = theta

    # X array , in/outputs tensor
    def error(self, X, inputs, outputs, beg, end):
        values = self.func(torch.tensor(X), self.theta, beg, end)
        if bkd.ndim(values) > 0 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC func should return an array of shape N by 1 for a single"
                " component. Use argument 'component' for different components."
            )
        return outputs[beg:end, self.component: self.component + 1] - values


class ParametricPointSetBC(object):
    """Dirichlet boundary condition for a set of points.
        Compare the output (that associates with `points`) with `values` (target data).

        Args:
            points: An array of points where the corresponding target values are known and used for training.
            values: An array of values that gives the exact solution of the problem.
            component: The output component satisfying this BC.
        """

    def __init__(self, points, values, component=0, aux_fun=None):
        self.points = points  # tensorized
        self.points_th = torch.tensor(points)
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "PointSetBC should output 1D values. Use argument 'component' for different components."
            )
        if aux_fun is None:
            self.aux_fun = lambda _, x: x
        else:
            self.aux_fun = aux_fun
        self.theta = values
        self.component = component

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component: self.component + 1] - self.aux_fun(self.points_th, self.theta)


'''
    Operator BC on a discrete set of points
'''


class OperatorPointSetBC(object):
    """Dirichlet boundary condition for a set of points.
    Compare the output (that associates with `points`) with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are known and used for training.
        func : func(x, u, X)
        resample_func: current no parameters
    """

    def __init__(self, points, func, resample_func=None):
        self.points = np.array(points, dtype=config.real(np))
        self.func = func
        self.resample_func = resample_func

    def collocation_points(self, X):
        return self.points if self.resample_func is None else self.resample_func()

    def error(self, X, inputs, outputs, beg, end):
        return self.func(inputs[beg: end], outputs[beg:end], X[beg: end])


def parametric_npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for dde.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, theta, beg, end):
        return func(X[beg:end], theta)

    @wraps(func)
    def wrapper_cache(X, beg, end):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    return wrapper_nocache
