# Neural kernel network wrapper implementation using GPflow 2 and TensorFlow 2.
# This module is created by modifying "neural_kernel_network_wrapper.py" (original code) in
# https://github.com/ssydasheng/GPflow-Slim/blob/master/gpflowSlim/neural_kernel_network/neural_kernel_network_wrapper.py
#
# Implementation of forward passes are changed from the original code.
#
#----------------------------------------------------------------------------
# Copyright 2018 Shengyang Sun
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import positive

import tensorflow as tf
import numpy as np
import math
import sympy as sp

# DTYPE is set to the default_float using in GPflow
DTYPE=gpflow.config.default_float()

class NKNWrapper(object):
    def __init__(self, hparams:dict):
        self._LAYERS = dict(
            Linear=Linear,
            Product=Product,
            Activation=Activation)
        self._build_layers(hparams)

    def _build_layers(self, hparams):
        self._layers = [self._LAYERS[l['name']](**l['params']) for l in hparams]

    def forward(self, input):
        outputs = input  # shape = (shape of K or K_diag, input_dim)
        for l in self._layers:
            # Shape is transformed from (shape of K or K_diag, input_dim)
            # to (shape of K or K_diag, output_dim) in each layer.
            outputs = l.forward(outputs)
        outputs = tf.squeeze(outputs, axis=-1)
        return outputs

    @property
    def parameters(self):
        params = ()
        for l in self._layers:
            params += l.parameters
        return params

    def symbolic(self):
        ks = sp.symbols(['k'+str(i) for i in range(self._layers[0].input_dim)]) + [1.]
        for l in self._layers:
            ks = l.symbolic(ks)
        assert len(ks) == 1, 'output of NKN must only have one term.'
        return ks[0]


class _KernelLayer(object):
    def __init__(self, input_dim, name):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X):
        assert X.dim == 2, 'Input to KernelLayer must be 2-dimensional.'
        self.forward(X)

    def forward(self, input):
        raise NotImplementedError

    @property
    def parameters(self):
       raise NotImplementedError

    def symbolic(self, ks):
        """
        return symbolic formula for the layer
        :param ks: list of symbolic numbers
        :return: list of symbolic numbers
        """
        raise NotImplementedError


class Linear(_KernelLayer):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b` with
    positive weight and bias.
    """

    def __init__(self, input_dim, output_dim, name='Linear'):
        super(Linear, self).__init__(input_dim, name=name)
        self.output_dim = output_dim

        min_w, max_w = 1. / (2 * input_dim), 3. / (2 * input_dim)
        weights = np.random.uniform(
            low=min_w, high=max_w, size=[output_dim, input_dim]).astype(DTYPE)
        self._weights = gpflow.Parameter(
            weights, transform=positive(), name='weights')
        self._bias = gpflow.Parameter(
            0.01*np.ones([self.output_dim], dtype=DTYPE),
            transform=positive(), name='bias')

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def forward(self, input):
        # input shape = (N, N, input_dim)
        # output shape = (N, N, output_dim)
        res = tf.tensordot(
            input,
            tf.transpose(self.weights),
            axes=[[-1], [0]]) + self.bias
        return res

    @property
    def parameters(self):
        return (self._weights, self._bias)

    def symbolic(self, ks):
        out = []
        for i in range(self.output_dim):
            sum = self.bias.numpy()[i]
            w = self.weights.numpy()
            for j in range(self.input_dim):
                sum += ks[j] * w[i, j]
            out.append(sum)
        return out


class Product(_KernelLayer):
    """
    Takes kernel product on the nodes.
    """
    def __init__(self, input_dim, step, name='Product'):
        super(Product, self).__init__(input_dim, name=name)
        assert isinstance(step, int) and step > 1, 'step must be number greater than 1.'
        assert int(math.fmod(input_dim, step)) == 0, 'input dim must be multiples of step.'
        self.step = step

    def forward(self, input):
        # input shape = (N, N, input_dim)
        input_shape = tf.shape(input)
        # reshape to (N, N, input_dim/step, step)
        reshaped_shape = tf.concat(
            [input_shape[:-1],
            [self.input_dim//self.step, self.step]],
            axis=0
        )
        return tf.reduce_prod(
            tf.reshape(input, reshaped_shape), -1)

    @property
    def parameters(self):
        return ()

    def symbolic(self, ks):
        out = []
        for i in range(int(self.input_dim / self.step)):
            out.append(np.prod(ks[i*self.step:(i+1)*self.step]))
        return out


class Activation(_KernelLayer):
    def __init__(self, input_dim, activation_fn, activation_fn_params, name='Activation'):
        super(Activation, self).__init__(input_dim, name=name)
        self.activation_fn = activation_fn
        self.output_dim = input_dim
        self._parameters = activation_fn_params

    def forward(self, input):
        return self.activation_fn(input)

    @property
    def parameters(self):
        return self._parameters

    def symbolic(self, ks):
        return [self.activation_fn(k) for k in ks]