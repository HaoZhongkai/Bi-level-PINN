import torch
import torch.nn as nn

from .nn import NN
from .. import activations
from .. import initializers
from ... import config


class PFNN(NN):
    """Fully-connected neural network.
        Input example [4, [[64] * 4 ] * 4, 4]
    """

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super(PFNN, self).__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.layers = nn.ModuleList()

        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1] if isinstance(layer_sizes[-1], int) else len(layer_sizes[-1])

        assert self.output_dim == len(layer_sizes) - 2

        for i in range(1, len(layer_sizes) - 1):
            if isinstance(layer_sizes[-1], int):
                layer_size = [layer_sizes[0]] + layer_sizes[i] + [1]
            else:
                layer_size = [layer_sizes[0]] + layer_sizes[i] + [layer_sizes[-1][i - 1]]

            linears = nn.ModuleList()
            for j in range(len(layer_size) - 1):
                linears.append(nn.Linear(layer_size[j], layer_size[j + 1], dtype=config.real(torch)))
                initializer(linears[-1].weight)
                initializer_zero(linears[-1].bias)
            self.layers.append(linears)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        x_ = []
        for linears in self.layers:
            x_i = x
            for linear in linears[:-1]:
                x_i = self.activation(linear(x_i))
            x_i = linears[-1](x_i)

            x_.append(x_i)

        x_ = torch.cat(x_, dim=1)
        if self._output_transform is not None:
            x_ = self._output_transform(inputs, x_)
        return x_
