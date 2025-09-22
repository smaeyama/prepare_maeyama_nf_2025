# Neural kernel network implementation using GPflow 2 and TensorFlow 2.
# This module is created by modifying "neural_kernel_network.py" in
# https://github.com/ssydasheng/GPflow-Slim/blob/master/gpflowSlim/neural_kernel_network/neural_kernel_network.py
#
# Checking method of input NKN and hyperparameter initialization method are added
# to the original code.
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
from typing import List
from numpy.typing import NDArray
import tensorflow as tf
import gpflow
import sympy as sp

from .spectral_mixture import SpectralMixtureKernel
from .neural_kernel_network_wrapper import NKNWrapper
from .utils import spectral_mixture_Sun_et_al

from ..utils import check_neural_kernel_network, initialize_length_scales

_KERNEL_DICT=dict(
    White=gpflow.kernels.White,
    Constant=gpflow.kernels.Constant,
    ExpQuad=gpflow.kernels.RBF,
    RBF=gpflow.kernels.RBF,
    Matern12=gpflow.kernels.Matern12,
    Matern32=gpflow.kernels.Matern32,
    Matern52=gpflow.kernels.Matern52,
    Cosine=gpflow.kernels.Cosine,
    ArcCosine=gpflow.kernels.ArcCosine,
    Linear=gpflow.kernels.Linear,
    Periodic=gpflow.kernels.Periodic,
    RatQuad=gpflow.kernels.RationalQuadratic,
    SpectralMixture=SpectralMixtureKernel,  # Spectral Mixture Kernel implementation in mfmodeling-SingleGP.
    SpectralMixture_Sun_et_al=spectral_mixture_Sun_et_al  # Spectral Mixture Kernel by Sun et al.
)

class NeuralKernelNetwork(gpflow.kernels.Kernel):
    def __init__(self,
                 primitive_kernels: List[dict],
                 neural_network: List[dict],
                 **kwargs):
        """
        GPflow 2 implementation of Neural Kernel Network based on Sun et al.'s code (original code):
        https://github.com/ssydasheng/GPflow-Slim/blob/master/gpflowSlim/neural_kernel_network/neural_kernel_network.py
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument for NeuralKernelNetwork: {kwarg}")
        super(NeuralKernelNetwork, self).__init__(**kwargs)

        assert len(primitive_kernels) > 0, 'At least one kernel should be provided.'
        self.neural_network = neural_network
        self.primitive_kernels = primitive_kernels

        self._check_neural_kernel_network()

        # Set active_dims, if provided.
        active_dims = kwargs.get("active_dims")
        if active_dims is not None:
            for kernel_idx, _ in enumerate(primitive_kernels):
                # If kernel is SpectralMixture_Sun_et_al, set active_dims key to
                # 'params' for spectral_mixture_Sun_et_al.
                if primitive_kernels[kernel_idx]['name'] == 'SpectralMixture_Sun_et_al':
                    for sm_param_idx, _ in enumerate(primitive_kernels[kernel_idx]['params']['params']):
                        primitive_kernels[kernel_idx]['params']['params'][sm_param_idx]['active_dims'] = active_dims
                elif primitive_kernels[kernel_idx]['name'] != 'Periodic':
                    primitive_kernels[kernel_idx]['params']['active_dims'] = active_dims

        self._primitive_kernels = [
            _KERNEL_DICT[k['name']](**k.get('params', {})) for k in primitive_kernels]

        self._nknWrapper = NKNWrapper(neural_network)

        self._nkn_parameters = self._nknWrapper.parameters

    def _check_neural_kernel_network(self):
        """
        Check if neural network is set correctly.
        This method is added to the original code.
        """
        check_neural_kernel_network(
            self.neural_network,
            self.primitive_kernels)

    def initialize_hyperparameters(self, X: NDArray, Y: NDArray):
        """
        Random initialization of length scales and periods in primitive kernels.
        This method is added to the original code.
        """
        # Select active dimensions.
        X_active = X[..., self.active_dims]

        for i, kernel in enumerate(self._primitive_kernels):
            if hasattr(kernel, 'lengthscales'):
                ls = initialize_length_scales(X_active)
                self._primitive_kernels[i].lengthscales = gpflow.Parameter(
                    ls, transform=self._primitive_kernels[i].lengthscales.transform)
            if hasattr(kernel, 'period'):
                ls = initialize_length_scales(X_active)
                self._primitive_kernels[i].period = gpflow.Parameter(
                    ls, transform=self._primitive_kernels[i].period.transform)
            if hasattr(kernel, 'base_kernel'):
                ls = initialize_length_scales(X_active)
                self._primitive_kernels[i].base_kernel.lengthscales = gpflow.Parameter(
                    ls, transform=self._primitive_kernels[i].base_kernel.lengthscales.transform)
            if hasattr(kernel, 'initialize_hyperparameters'):  # Spectral Mixture case
                self._primitive_kernels[i].initialize_hyperparameters(X, Y)
            if hasattr(kernel, 'kernels'):    # Sun et al.'s Spectral Mixture case
                # Since Sun et al.'s Spectral Mixture is summation of kernel products,
                # whose components can be accessed via kernel.kernels.kernels
                for j, kernel_summand in enumerate(kernel.kernels):
                    for k, kernel in enumerate(kernel_summand.kernels):
                        if hasattr(kernel, 'lengthscales'):
                            ls = initialize_length_scales(X_active)
                            self._primitive_kernels[i].kernels[j].kernels[k].lengthscales \
                                = gpflow.Parameter(
                                    ls,
                                    transform=self._primitive_kernels[i].kernels[j].kernels[k].lengthscales.transform)


    # Decorate with the tf.function decorator for faster execution
    @tf.function
    def K_diag(self, X):
        # Series of  {K_diag^i(X, X2)}_{i=1, ..., k}, shape = (N, k) where N and k stand for
        # number of samples and primitive kernels, respectively.
        primitive_values = tf.stack(
            [kern.K_diag(X)
             for kern in self._primitive_kernels],
            axis=-1)
        return self._nknWrapper.forward(primitive_values)

    # Decorate with the tf.function decorator for faster execution
    @tf.function
    def K(self, X, X2=None):
        # Series of  {K_diag^i(X, X2)}_{i=1, ..., k}, shape = (N, N, k) where N and k stand for
        # number of samples and primitive kernels, respectively
        primitive_values = tf.stack(
            [kern.K(X, X2)
             for kern in self._primitive_kernels],
            axis=-1)
        return self._nknWrapper.forward(primitive_values)

    @property
    def polynomial(self):
        """
        Polynomial of the neural network.
        Added to the original code.
        """
        return sp.expand(self._nknWrapper.symbolic())