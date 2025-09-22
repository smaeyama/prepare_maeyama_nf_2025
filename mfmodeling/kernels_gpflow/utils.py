"""
Subfunctions used in kernel settings.
This module depends on GPflow.
"""
import gpflow
from typing import Optional, Sequence, Union, List
from numpy.typing import NDArray

from mfmodeling.utils import length_scale_Sun_et_al

ActiveDims = Union[slice, Sequence[int]]


def count_n_features(
        X: NDArray,
        active_dims: Optional[ActiveDims] = None):
    """
    Count the number of features from input data and active_dims.

    Parameters
    ----------
    X : NDArray
        Data.
    active_dims : ActiveDims
        Active dimension.

    Returns
    ----------
    n_features : Int
        Number of features.
    """
    if active_dims is None:
        n_features = X.shape[1]
        return n_features

    if isinstance(active_dims, slice):
        if active_dims.step is None:
            n_features = active_dims.stop - active_dims.start
        else:
            n_features = (
                (active_dims.stop - active_dims.start
                    + active_dims.step - 1) // active_dims.step
            )
    else:
        n_features = len(active_dims)

    return n_features


def spectral_mixture_Sun_et_al(params: List[dict]):
    """
    Builds the Spectral Mixture kernel.
    This function is a GPflow 2 implementation of Spectral Mixture Kernel
    written by Sun et al.:
    https://github.com/ssydasheng/Neural-Kernel-Network/blob/master/kernels.py

    The code in the above link is modified.

    Parameters
    ---------
    params: list[Dict]
        With each item corresponding to one mixture.
        The dict is formatted as {'w': float, 'rbf': dict, 'cos': dict}.
        That each sub-dict is used to the init corresponding kernel.

    Returns
    ---------
    gpflow.kernels.Kernel
        Spectral mixture kernel object.
    """
    rbf = gpflow.kernels.RBF(**params[0].get('rbf', {}))
    rbf.variance = gpflow.Parameter(1.0, trainable=False)  # Fix variance to be 1.0
    cosine = gpflow.kernels.Cosine(**params[0].get('cos', {}))
    cosine.variance = gpflow.Parameter(1.0, trainable=False)  # Fix variance to be 1.0
    constant = gpflow.kernels.Constant(**params[0].get('w', {}))  # Corresponding to weights
    sm = constant * rbf * cosine
    for i in range(1, len(params)):
        rbf = gpflow.kernels.RBF(**params[i].get('rbf', {}))
        rbf.variance = gpflow.Parameter(1.0, trainable=False)
        cosine = gpflow.kernels.Cosine(**params[i].get('cos', {}))
        cosine.variance = gpflow.Parameter(1.0, trainable=False)
        constant = gpflow.kernels.Constant(**params[i].get('w', {}))
        sm += constant * rbf * cosine
    return sm


def set_default_kernel_args(
        kernel_name: str,
        X: NDArray,
        active_dims: ActiveDims):
    """
    Setting default kernel arguments using input data.

    Parameters
    ---------
    kernel_name : Str
        Kernel name.
    X : NDArray
        Reference data.
    active_dims : ActiveDims
        Active dimension.


    Returns
    ---------
    Dict
        Default setting for specified kernel.
    """
    n_features = count_n_features(X, active_dims)

    if kernel_name == "RBF":
        return {"lengthscales": [1.0]*n_features, "variance": 1.0}
    elif kernel_name == "SpectralMixture":
        return {"n_components": 10}
    elif kernel_name == "NeuralKernelNetwork":
        if active_dims is not None:
            X = X[..., active_dims]

        ls = length_scale_Sun_et_al(X)

        primitive_kernels = [
            {'name': 'Linear',   'params': {'active_dims': active_dims}},
            {'name': 'Periodic', 'params': {
                'base_kernel' : gpflow.kernels.SquaredExponential(
                    lengthscales=ls, active_dims=active_dims), 'period': ls}},
            {'name': 'ExpQuad',  'params': {'lengthscales': ls / 4.0, 'active_dims': active_dims}},
            {'name': 'RatQuad',  'params': {'alpha': 0.2, 'lengthscales': ls * 2.0, 'active_dims': active_dims}},
            {'name': 'Linear',   'params': {'active_dims': active_dims}},
            {'name': 'RatQuad',  'params': {'alpha': 0.1, 'lengthscales': ls, 'active_dims': active_dims}},
            {'name': 'ExpQuad',  'params': {'lengthscales': ls, 'active_dims': active_dims}},
            {'name': 'Periodic', 'params': {
                'base_kernel' : gpflow.kernels.SquaredExponential(
                    lengthscales=ls / 4.0, active_dims=active_dims), 'period': ls / 4.0}}]
        neural_network = [
                {'name': 'Linear',  'params': {'input_dim': 8, 'output_dim': 8,}},
                {'name': 'Product', 'params': {'input_dim': 8, 'step': 2}},
                {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4}},
                {'name': 'Product', 'params': {'input_dim': 4, 'step': 2}},
                {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1}}]
        return {"primitive_kernels": primitive_kernels,
                "neural_network": neural_network}
