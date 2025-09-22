"""
Subfunctions used in mfmodeling package.
This module contains functions that are independent
of the Gaussian process libraries used.
"""
import warnings
from typing import Sequence, Union, List, Dict
import math
import numpy as np
from numpy.typing import NDArray

ActiveDims = Union[slice, Sequence[int]]


def normalize_data(X: NDArray, normalizer: NDArray):
    """
    Normalize array along data axis.

    Parameters
    ----------
    X : Numpy.ndarray[ndata, ninput]
        Data to normalize.
    normalizer : Numpy.ndarray[ndata, ninput]
        The normalizer.

    Returns
    -------
    X_normalized : Numpy.ndarray[ndata, ninput]
        Normalized data.
    """
    mean = np.mean(normalizer, axis=0)
    std = np.std(normalizer, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized


def denormalize_data(X: NDArray, denormalizer: NDArray):
    """
    Denormalize array along data axis.

    Parameters
    ----------
    X : Numpy.ndarray[ndata, ninput]
        Data to denormalize.
    denormalizer : Numpy.ndarray[ndata, ninput]
        The denormalizer.

    Returns
    -------
    X_denormalized : Numpy.ndarray[ndata, ninput]
        Denormalized data.
    """
    mean = np.mean(denormalizer, axis=0)
    std = np.std(denormalizer, axis=0)
    X_denormalized = std * X + mean
    return X_denormalized


def normalize_data_list_columnwise(
        X_list: List[NDArray],
        normalizer_list: List[NDArray]):
    """
    Normalize each array contained in the list along the data axis.
    The normalizer is a list of data, and the data for each dimension
    is concatenated for all samples in the list.

    Parameters
    ----------
    X_list : list(Numpy.ndarray[ndata, ninput])
        Data list to normalize.
    normalizer_list : list(Numpy.ndarray[ndata, ninput])
        List of normalizers.

    Returns
    -------
    X_denormalized : Numpy.ndarray[ndata, ninput]
        Denormalized data.
    """
    # Add each array column to list
    normalizer_column_list = []
    for normalizer in normalizer_list:
        for i in range(normalizer.shape[1]):
            if len(normalizer_column_list) <= i:
                normalizer_column_list.append(normalizer[:, i])
            else:
                normalizer_column_list[i] = np.concatenate(
                    (normalizer_column_list[i], normalizer[:, i]))
    # Calculate the mean and std of each column
    mean_list = [np.mean(col) for col in normalizer_column_list]
    std_list = [np.std(col) for col in normalizer_column_list]

    # Normalize along each column
    X_normalized_list = []
    for X in X_list:
        X_normalized = np.empty(X.shape)
        for i in range(X.shape[1]):
            X_normalized[:, i] = (X[:, i] - mean_list[i]) / std_list[i]
        X_normalized_list.append(X_normalized)
    return X_normalized_list


def denormalize_mean(
        mean: NDArray,
        denormalizer: NDArray):
    """
    Denormalize mean array along data axis.
    This function wraps denormalize_data function.

    Parameters
    ----------
    mean : Numpy.ndarray[ndata, ninput]
        Data to normalize.
    denormalizer : Numpy.ndarray[ndata, ninput]
        The denormalizer.

    Returns
    -------
    Numpy.ndarray[ndata, ninput]
        Denormalized data.
    """
    return denormalize_data(mean, denormalizer)


def denormalize_variance(
        var: NDArray,
        denormalizer: NDArray):
    """
    Denormalize mean array along data axis.
    This function wraps denormalize_data function.

    Parameters
    ----------
    mean : Numpy.ndarray[ndata, ninput]
        Data to normalize.
    denormalizer : Numpy.ndarray[ndata, ninput]
        The denormalizer.

    Returns
    -------
    Numpy.ndarray[ndata, ninput]
        Denormalized mean.
    """
    return np.std(denormalizer, axis=0)**2 * var


def initialize_length_scales(X: NDArray):
    """
    Initialization of length scales.
    This function returns a random array
    between the minimum and median of the data distance.

    Parameters
    ----------
    X : Numpy.ndarray[ndata, ninput]
        Reference data.

    Returns
    -------
    Numpy.ndarray[ninput]
        Random array between the minimum
        and median of the data distance.
    """
    # Distance matrices
    dist_matrices = np.sqrt(np.square(np.expand_dims(X, 1) - X))
    masked_distances = np.ma.masked_equal(dist_matrices, 0)

    # Minimum non-zero data distances
    minimum_length_scales = np.ma.min(masked_distances, axis=(0, 1))

    # Median of non-zero data distances
    median_length_scales = np.ma.median(masked_distances, axis=(0, 1))

    return np.random.uniform(
        low=minimum_length_scales, high=median_length_scales)


def check_neural_kernel_network(
        neural_network : List[Dict],
        primitive_kernels : List[Dict]):
    """
    Method to to verify the integrity of NKN.
    The following conditions are checked if satisfied.
    * params and name keys are set for each layer.
    * Input dimension of the first layer is same as the number of primitive kernels.
    * Input dimension of a layer is same as output dimension of the last layer.
    * Output dimension of the last layer is 1.

    Parameters
    ----------
    neural_network : Dict
        Neural network used in NKN.
    neural_network : Dict
        Primitive kernels used in NKN.
    """
    if not neural_network:
        return
    last_dim = len(primitive_kernels)
    for i, layer in enumerate(neural_network):
        params = layer.get('params')
        if params is None:
            # If params key does not exist, raise
            raise KeyError(f"params key for layer{i} is not set.")
        layer_name = layer.get('name')

        if layer_name is None:
            # If params key does not exist, raise
            raise KeyError(f"name key for layer{i} is not set.")

        if layer_name == 'Linear':
            are_io_dimensions_ok = check_linear_dimensions(params, last_dim)
            if not are_io_dimensions_ok:
                raise ValueError(f"Invalid dimensions are set for layer{i} (Linear layer).")
            last_dim = params.get('output_dim')
        elif layer_name == 'Product':
            are_io_dimensions_ok = check_product_dimensions(params, last_dim)
            if not are_io_dimensions_ok:
                raise ValueError(f"Invalid input dimension or step are set for layer{i} (Product layer).")
            last_dim = params.get('input_dim')//params.get('step')  # output_dim = input_dim/step
        elif layer_name == 'Activation':
            are_io_dimensions_ok = check_activation_dimension(params, last_dim)
            if not are_io_dimensions_ok:
                raise ValueError(f"Invalid input dimension set for layer{i} (Activation layer).")
            last_dim = params.get('input_dim')  # output_dim = input_dim
        else:
            raise ValueError(f"Unknown layer is set for layer{i}.")
    # The last output dimension must be 1
    if last_dim != 1:
        raise ValueError(f"The last output dimension must be 1.")


def check_linear_dimensions(
        layer: dict,
        last_dim: int):
    """
    Checks if input and output dimensions of
    a Linear layer are set correctly.

    Parameters
    ----------
    layer : Dict
        Current layer.
    last_dim : Dict
        Output dimensions of last layer.

    Returns
    ----------
    Bool
        Returns True if input and output dimensions
        are set correctly.
    """
    input_dim = layer.get('input_dim')
    if input_dim != last_dim:
        return False
    output_dim = layer.get('output_dim')
    if output_dim is None:
        return False
    return True


def check_product_dimensions(
        layer: dict,
        last_dim: int):
    """
    Checks if input and output dimensions of
    a Product layer are set correctly.

    Parameters
    ----------
    layer : Dict
        Current layer.
    last_dim : Dict
        Output dimensions of last layer.

    Returns
    ----------
    Bool
        Returns True if input and output dimensions
        are set correctly.
    """
    input_dim = layer.get('input_dim')
    step = layer.get('step')
    if input_dim != last_dim:
        return False
    if step is None:
        return False
    if math.fmod(input_dim, step) != 0:
        return False
    return True


def check_activation_dimension(
        layer: dict,
        last_dim: int):
    input_dim = layer.get('input_dim')
    """
    Checks if input and output dimensions of
    an Activation layer are set correctly.

    Parameters
    ----------
    layer : Dict
        Current layer.
    last_dim : Dict
        Output dimensions of last layer.

    Returns
    ----------
    Bool
        Returns True if input and output dimensions
        are set correctly.
    """
    if input_dim is None:
        warnings.warn(
            "Input dimension of activation layer is set to be {last_dim}.")
    elif input_dim != last_dim:
        return False
    return True


def length_scale_Sun_et_al(X: NDArray, normalize: bool = False):
    """
    Gets the median of distances between X.
    This function is taken from the following:
    https://github.com/ssydasheng/Neural-Kernel-Network/blob/master/utils/functions.py

    Parameters
    ----------
    X : Numpy.ndarray[ndata, ninput]
        Reference data.
    normalize : Bool
        Normalize data if True.

    Returns
    -------
    Numpy.ndarray[ninput]
        Median of distances between X.
    """
    if normalize:
        x = normalize_data(X, X)
    else:
        x = X
    if X.shape[0] > 10000:
        permutation = np.random.permutation(X.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * np.sqrt(X.shape[1])
