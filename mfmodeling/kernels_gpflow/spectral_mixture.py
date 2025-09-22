from typing import Optional
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.utilities import positive
from check_shapes import inherit_check_shapes


class SpectralMixtureKernel(gpflow.kernels.Kernel):
    """
    Spectral Mixture Kernel based on the following paper [Wilson and Adams].
    Wilson, Andrew, and Ryan Adams.,
    "Gaussian process kernels for pattern discovery and extrapolation.",
    International conference on machine learning. PMLR, 2013.

    This implementation is based on the corrected version of
    Eq. (12) from the above paper, as shown in Eq. (5) of the following PDF.

    https://www.cs.cmu.edu/~andrewgw/typo.pdf
    """
    def __init__(self,
                 n_features: int, n_components: int,
                 mean_list: Optional[NDArray] = None,
                 variance_list: Optional[NDArray] = None,
                 weight_list: Optional[NDArray] = None,
                 **kwargs):
        """
        Parameters
        ----------
        n_features : Int
            Number of features. "P" in [Wilson and Adams]
        n_components : Int
            Number of spectral components. "Q" in [Wilson and Adams]
        mean_list : Optional[NDArray]
            Initial mean vectors in the cosine term.
        variance_list : Optional[NDArray]
            Initial variances in the RBF term.
        weight_list : Optional[NDArray]
            Initial weights for each spectral components.
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_components = n_components

        # Setting of data type
        self.dtype = gpflow.config.default_float()
        self.int = gpflow.config.default_int()

        if mean_list is not None:
            if mean_list.shape == (n_features, n_components):
                self.mean_list = gpflow.Parameter(
                    mean_list, transform=positive())
            else:
                raise ValueError(
                    "SpectralMixtureKernel: Dimension of given mean_list does not match n_features.")
        else:
            self.mean_list = gpflow.Parameter(
                tf.ones((n_features, n_components)), transform=positive())

        if variance_list is not None:
            if variance_list.shape == (n_features, n_components):
                self.variance_list = gpflow.Parameter(
                    variance_list, transform=positive())
            else:
                raise ValueError(
                    "SpectralMixtureKernel: Dimension of given variance_list does not match n_features.")
        else:
            self.variance_list = gpflow.Parameter(
                tf.ones((n_features, n_components)), transform=positive())

        if weight_list is not None:
            if weight_list.shape == (n_components, ):
                self.weight_list = gpflow.Parameter(
                    weight_list, transform=positive())
            else:
                raise ValueError(
                    "SpectralMixtureKernel: Dimension of given variance_list does not match n_features.")
        else:
            self.weight_list = gpflow.Parameter(
                tf.ones(n_components), transform=positive())
        # Validate active_dims
        self._validate_nfeature_active_dims()

    def _validate_nfeature_active_dims(self):
        """
        Validate that n_features parameter matches the number of active_dims.
        """
        if isinstance(self.active_dims, slice) and self.active_dims == slice(None, None, None):
            return

        if isinstance(self.active_dims, slice):
            if self.active_dims.step is None:
                active_dims_length = self.active_dims.stop - self.active_dims.start
            else:
                active_dims_length = (
                    (self.active_dims.stop - self.active_dims.start
                     + self.active_dims.step - 1) // self.active_dims.step
                )
        else:
            active_dims_length = len(self.active_dims)

        if self.n_features != active_dims_length:
            raise ValueError(
                f"Size of `active_dims` {active_dims_length} does not match "
                f"n_features ({self.n_features})."
            )

    def initialize_hyperparameters(self, X, Y):
        """
        Initialization of the mean, variance and weight lists.
        The implementation is based on "initSMhypers.m"
        https://people.orie.cornell.edu/andrew/code/initSMhypers.m

        #----------------------------------------------------------------------------
        % Initialisation script for SM kernel.
        % Andrew Gordon Wilson, 11 Oct 2013

        % Example hyper initialisation for spectral mixture kernel.
        % One can improve this initialisation scheme by sampling from (or
        % projecting) the empirical spectral density.

        % This is not an all purpose initialisation script.  Common sense is
        % still required when initialising the SM kernel in new situations.
        #----------------------------------------------------------------------------
        """
        # Select active dimensions
        X = X[..., self.active_dims]
        n_samples = X.shape[0]

        # Initialize weight_list
        self.weight_list.assign(
            tf.math.reduce_std(Y) / self.n_components
            * tf.ones(self.n_components, dtype=self.dtype))

        # sqrt((X_{pd} - X_{qd})^2_{pqd})
        dist_matrices = tf.math.sqrt(tf.square(tf.expand_dims(X, 1) - X))
        non_zero_mask = tf.not_equal(dist_matrices, 0)
        zero_mask = tf.logical_not(non_zero_mask)

        max_value = tf.reduce_max(dist_matrices)
        dist_matrices_for_minimum = tf.zeros_like(dist_matrices)
        dist_matrices_for_minimum = tf.where(
            non_zero_mask,
            dist_matrices,
            dist_matrices_for_minimum)
        dist_matrices_for_minimum = tf.where(
            zero_mask,
            max_value,
            dist_matrices_for_minimum)

        # Calculate the Nyquist frequency
        if n_samples == 1:
            min_shift = tf.ones(self.n_features, dtype=self.dtype)
            max_shift = tf.ones(self.n_features, dtype=self.dtype)
        else:
            min_shift = tf.reduce_min(
                dist_matrices_for_minimum,
                axis=[0, 1])
            max_shift = tf.reduce_max(
                dist_matrices,
                axis=[0, 1])
        nyquist = tf.tile(
            tf.expand_dims(0.5 / min_shift, 1),
            [1, self.n_components])
        # Uniform random tensor of n_feature * n_components
        uniform_rand = tf.random.uniform(
            (self.n_features, self.n_components),
            dtype=self.dtype)
        self.mean_list.assign(nyquist * uniform_rand)

        # Normal random tensor of n_feature * n_components
        normal_rand = tf.random.normal(
            (self.n_features, self.n_components),
            dtype=self.dtype)
        inverse_std = tf.tile(
            tf.expand_dims(max_shift, 1),
            [1, self.n_components]
        )
        self.variance_list.assign(
            tf.math.abs(
                (1.0 / normal_rand / inverse_std)**2
            ))

    # "inherit_check_shapes" decorator is inherited from
    # "check_shape" decorator of the super class, which checks
    # the arguments and return values of K are following shapes:
    #    "X: [batch..., N, P]",
    #    "X2: [batch2..., N2, P]",
    #    "return: [batch..., N, batch2..., N2] if X2 is not None",
    #    "return: [batch..., N, N] if X2 is None",
    @inherit_check_shapes
    def K(self, X, X2=None):
        """
        Override of the abstract method K.
        """
        if X2 is None:
            X2 = X

        # Difference matrix tau = {|x_p -x'_p}}_sample
        # shape = [batch..., N, N, P]
        tau = gpflow.utilities.ops.difference_matrix(X, X2)

        # tau_p^2 shape = [batch..., N, N, P]
        tau2 = tau * tau

        # sum_{p=1}^P tau_p^2 v_q^(p), shape = [batch..., N, N, P]
        # here, variance_list.shape = [P, Q]
        tau2_mul_v = tf.tensordot(
                tau2, self.variance_list, axes=[[-1], [0]])

        # Exponent of the exponential term, shape = [batch..., N, N, Q]
        exponents = -2 * np.pi * np.pi * tau2_mul_v

        # sum_{p=1}^P tau_p * mu_q^{(p)}
        # tau.shape = [batch..., N, N, P]
        tau_mul_mu = tf.tensordot(tau, self.mean_list, axes=[[-1], [0]])  # shape = [batch..., N, N, Q]

        # Cosine term, shape = [batch..., N, N, Q].
        cosines = tf.math.cos(2.0 * np.pi * tau_mul_mu)

        # Calculate the spectral mixture kernel
        # Here, weight_list.shape = [Q]
        k = tf.tensordot(self.weight_list,
                         tf.math.multiply(tf.math.exp(exponents), cosines),
                         axes=[[0], [-1]])
        return k

    # "inherit_check_shapes" decorator is inherited from
    # "check_shape" decorator of the super class, which checks
    # the arguments and return values of K are following shapes:
    #    "X: [batch..., N, P]",
    #    "return: [batch..., N]"
    @inherit_check_shapes
    def K_diag(self, X):
        """
        Override of the abstract method K_diag.
        """
        return tf.fill(
            tf.shape(X)[:-1], tf.reduce_sum(self.weight_list))