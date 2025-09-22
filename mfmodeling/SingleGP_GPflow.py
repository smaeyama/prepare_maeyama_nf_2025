#!/usr/bin/env python
# coding: utf-8

import copy
import warnings
from typing import Optional
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from gpflow.models import GPR

from .kernels_gpflow.spectral_mixture import SpectralMixtureKernel
from .kernels_gpflow.neural_kernel_network import NeuralKernelNetwork
from .kernels_gpflow.utils import count_n_features, set_default_kernel_args
from .utils import normalize_data, denormalize_data


class SingleGP():
    '''
    Gaussian process regression.
    '''
    # Initialize number of features
    __n_features = 0

    # Dictionary of kernels and their default variables
    KERNEL_DICT = {
        "RBF": gpflow.kernels.SquaredExponential,
        "SpectralMixture": SpectralMixtureKernel,
        "NeuralKernelNetwork": NeuralKernelNetwork
    }

    @property
    def __default_kernel_args(self):
        """
        Setting default kernel arguments using input data.
        """
        return set_default_kernel_args(
            kernel_name=self.__kernel_name,
            X=self.__data[0],
            active_dims=self.__kernel_args.get("active_dims"))

    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        data : list[2]
            List of training data of y=f(x), where data input x and output y are vectors.
            [data_input[nsample,ninput], data_output[nsample,noutput]]
        kernel : Str
            Name of kernel to use.
        kernel_args : Dict
            Optional arguments for the kernel.
        noise_level : Float
            Optional initial value for the noise level.
        normalize : Bool
            Options for normalizing input data.
        augmented_dim : Int
            Augmented dimension used for create the NARGP kernel.
            It this parameter is set, NARGP kernel is created.
        '''
        # Check key word arguments
        for kwarg in kwargs:
            if kwarg not in {"data",
                             "kernel",
                             "kernel_args",
                             "noise_level",
                             "normalize",
                             "augmented_dim"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        # (List) training data
        if kwargs.get("normalize") is None:
            self.__normalize = False
        else:
            self.__normalize = kwargs.get("normalize")

        # Check precision
        data = kwargs.get("data")
        self.__dtype = data[0].dtype

        # Set default dtype in gpflow
        gpflow.config.set_default_float(self.__dtype)

        # Normalization
        self.__X_normalizer = data[0]
        self.__Y_normalizer = data[1]
        if self.__normalize:
            X = data[0]
            Y = data[1]
            self.__data = [
                normalize_data(X, self.__X_normalizer),
                normalize_data(Y, self.__Y_normalizer)]
        else:
            self.__data = data

        # (string) kernel name, default to "RBF"
        self.__kernel_name = kwargs.get("kernel", "RBF")
        if self.__kernel_name not in self.KERNEL_DICT:
            # If unknown kernel is given, raise ValueError
            raise ValueError("SingleGP: Unknown kernel is specified.")

        # (dict) optional kernel arguments
        self.__kernel_args = kwargs.get("kernel_args")

        # If kernel arguments are not set or only active_dims is set,
        # use default value after counting input dimension
        self.__use_default_kernel_args = self.__kernel_args is None or self.__kernel_args == {}

        # Check if SingleGP is called for NARGP.
        self.__used_for_nargp = kwargs.get("augmented_dim") is not None

        if self.__used_for_nargp:
            self.__augmented_dim = kwargs.get("augmented_dim")
        else:
            if self.__use_default_kernel_args:
                # Count input dimension
                self.__kernel_args = {}
                self.__n_features = count_n_features(self.__data[0])
                self.__kernel_args = self.__default_kernel_args

            # Count input dimension.
            self.__n_features = count_n_features(
                self.__data[0],
                self.__kernel_args.get("active_dims"))

            if self.__kernel_name == "SpectralMixture":
                # If kernel is Spectral Mixture, number of features
                # (n_features) must be set
                self.__kernel_args["n_features"] = self.__n_features

        # (float) noise level. If not set, default to 0.01
        self.__noise_level = kwargs.get("noise_level", 0.01)

        # Initialization of GPR model
        self.__model = None

    def __set_optimizer(self,
                        optimizer_name, max_iters, optimizer_parameters):
        """
        Set the optimization method.
        Optimization method is returned as a function object.
        """
        # Default arguments for optimizers
        default_optimizer_args_dict = {
            "Adam": {"learning_rate": 0.001},
            "Scipy": {"method": "L-BFGS-B", "options": {"maxiter": max_iters}}
        }

        # Set optimization method
        if optimizer_name == "Adam":
            # Adam case
            if optimizer_parameters is None:
                # Default parameters
                optimizer_parameters = default_optimizer_args_dict["Adam"]

            def run_optimizer(model: GPR):
                opt = tf.optimizers.Adam(**optimizer_parameters)
                training_loss = model.training_loss_closure()

                @tf.function
                def train_step():
                    opt.minimize(training_loss, model.trainable_variables)

                for _ in range(max_iters):
                    train_step()
        elif optimizer_name == "Scipy":
            # Scipy case
            if optimizer_parameters is None:
                # Default parameters
                optimizer_parameters = default_optimizer_args_dict["Scipy"]
                optimizer_parameters["options"]["maxiter"] = max_iters
            else:
                # "maxiter" is overwritten by the input "max_iter"
                options = optimizer_parameters.get("options", {})
                options.setdefault("maxiter", max_iters)

            def run_optimizer(model: GPR):
                opt = gpflow.optimizers.Scipy()
                method = optimizer_parameters.get(
                    "method",
                    default_optimizer_args_dict["Scipy"]["method"])
                options = optimizer_parameters.get(
                    "options", {})
                opt.minimize(
                    model.training_loss,
                    model.trainable_variables,
                    method=method,
                    options=options)
        else:
            raise ValueError(
                "SingleGP.optimize: Unknown optimizer "
                + optimizer_name + " is specified."
                + "Please specify either 'Scipy' or 'Adams'.")
        return run_optimizer

    def __set_kernel(self):
        """
        Sets arguments for the kernels contained in KERNEL_DICT
        to create a kernel.
        """
        return self.KERNEL_DICT[self.__kernel_name](**self.__kernel_args)

    def __set_kernel_args_for_nargp(self, active_dims):
        """
        Set default kernel arguments for superposed kernel in NARGP.
        """
        if self.__use_default_kernel_args:
            kernel_args = {"active_dims": active_dims}
            kernel_args.update(set_default_kernel_args(
                kernel_name=self.__kernel_name,
                X=self.__data[0],
                active_dims=active_dims))
        else:
            kernel_args = self.__kernel_args
            kernel_args["active_dims"] = active_dims

        if self.__kernel_name == "SpectralMixture":
            # If kernel is Spectral Mixture, number of features (n_features)
            # must be set
            kernel_args["n_features"] = count_n_features(self.__data[0], active_dims)
        return kernel_args

    def __superposed_kernel_for_nargp(self):
        """
        Superposition of kernels used in NARGP.
        """
        ninput = self.__data[0].shape[1] - self.__augmented_dim
        noutput = self.__augmented_dim

        # k_rho
        k_rho_args = self.__set_kernel_args_for_nargp(
            active_dims=slice(0, ninput))
        k_rho = self.KERNEL_DICT[self.__kernel_name](**k_rho_args)

        # k_f
        k_f_args = self.__set_kernel_args_for_nargp(
            active_dims=slice(ninput, ninput + noutput))
        k_f = self.KERNEL_DICT[self.__kernel_name](**k_f_args)

        # k_delta
        k_delta_args = self.__set_kernel_args_for_nargp(
            active_dims=slice(0, ninput))
        k_delta = self.KERNEL_DICT[self.__kernel_name](**k_delta_args)

        return k_rho * k_f + k_delta

    def __design_kernel(self):
        """
        Design kernel. If this instance is used for NARGP,
        the superposed kernel in NARGP is returned.
        """
        if self.__used_for_nargp:
            return self.__superposed_kernel_for_nargp()
        else:
            return self.__set_kernel()

    def optimize(self, optimize_restarts=30, max_iters=400,
                 optimizer_name="Scipy",
                 optimizer_parameters: Optional[dict] = None,
                 verbose=False,
                 plot_history=False,
                 optimize_noise_in_first_step=False,
                 use_built_in_initialization=False):
        '''
        Optimization of the hyperparameters of the kernel
        and the noise variance.

        Parameters
        ----------
        optimize_restarts : Int
            Number of iterations in the optimize_restart method.
        max_iters : Int
            Maximum number of iterations in the optimization algorithm.
        optimizer_name : Str
            TensorFlow optimizer name. Either "Scipy" and "Adam", which use
            gpflow.optimizer.Scipy and tf.optimizer.Adam, respectively,
            are available. Default to "Scipy".
        optimizer_parameters: Dict
            Optional detailed optimization settings.
        verbose : Bool
            Shows progress of the __optimize_restart if True.
        plot_history : Bool
            Shows history of the objective function in the __optimize_restart if True.
        optimize_noise_in_first_step : Bool
            Optimize noise level in the first step if True.
        use_built_in_initialization : Bool
            Use hyper parameter initialization method (if any) of the kernel.
        '''
        if verbose:
            print(
                "optimize_restarts=", optimize_restarts,
                ", max_iters=", max_iters)

        # Single-fidelity GP
        X, Y = self.__data

        # Kernel design
        k = self.__design_kernel()

        # Create instance of GP Regression
        m = gpflow.models.GPR(data=(X, Y), kernel=k)

        # Initialization of the hyperparameters
        m.likelihood.variance = self.__noise_level * tf.math.reduce_variance(
            m.data[1])

        if optimize_noise_in_first_step:
            # Exclude the noise level from the optimization
            m.likelihood.variance = gpflow.Parameter(
                m.likelihood.variance, trainable=True,
                transform=gpflow.utilities.positive())
        else:
            # Exclude the noise level from the optimization
            m.likelihood.variance = gpflow.Parameter(
                m.likelihood.variance, trainable=False)

        # Set optimizers.
        run_optimizer = self.__set_optimizer(
            optimizer_name, max_iters, optimizer_parameters)

        # If use_built_in_initialization is True and kernel has its initialization method,
        # use initialization method of the kernel
        if (hasattr(m.kernel, "initialize_hyperparameters")
            and
            use_built_in_initialization):
            m.kernel.initialize_hyperparameters(X, Y)

        # Optimization of hyperparameters with fixed noise level
        run_optimizer(m)

        # Unfix the noise level to optimize
        m.likelihood.variance = gpflow.Parameter(
            m.likelihood.variance,
            trainable=True,
            transform=gpflow.utilities.positive())

        # Perform the optimize_restart
        m = self.__optimize_restart(
            m, run_optimizer,
            optimize_restarts,
            verbose,
            plot_history)

        self.__model = copy.deepcopy(m)
        self.__kernel = copy.deepcopy(m.kernel)
        return

    def predict(self, x):
        '''
        Parameters
        ----------
        x : Numpy.ndarray[..., ninput]
            The points at which to make a prediction.

        Returns
        -------
        mean : Numpy.ndarray[..., noutput]
        variance : Numpy.ndarray[..., noutput]
        '''
        # Convert x to self.__dtype
        if x.dtype != self.__dtype:
            warnings.warn(
                "Input dtype {x.dtype} of predict is different from using dtype. \
                    Convert to {self.__dtype}.")
            x = x.astype(self.__dtype)

        m = self.__model
        if self.__normalize:
            mu0, var0 = m.predict_y(
                normalize_data(x, self.__X_normalizer))
            mu0 = mu0.numpy()
            var0 = var0.numpy()
            mu0 = denormalize_data(mu0, self.__Y_normalizer)
            var0 = np.std(self.__Y_normalizer, axis=0)**2 * var0
        else:
            mu0, var0 = m.predict_y(x)
            mu0 = mu0.numpy()
            var0 = var0.numpy()
        return mu0, var0

    def predict_f_samples(self, x, nsamples):
        """
        Produce samples from the posterior latent function(s) at the input points.
        This method wraps GPflow predict_f_samples.
        The shape of the return value will be transformed
        to match the shape of the return value of GPy posterior_samples_f.

        Parameters
        ----------
        x : Numpy.ndarray[..., ninput]
            The points at which to make a prediction.
        nsamples : Int
            Number of samples to draw.

        Returns
        -------
        z_samples : Numpy.ndarray[..., ndata, noutput, nsamples]
        """
        def __gpflow_format_to_gpy_format(z_samples):
            """
            Convert shape of return value of GPflow "predict_f_samples"
            to that of GPy "posterior_samples_f".
            """
            return tf.transpose(z_samples, perm=[1, 2, 0]).numpy()

        # Convert x to self.__dtype
        if x.dtype != self.__dtype:
            warnings.warn(
                "Input dtype {x.dtype} of predict is different from using dtype. \
                    Convert to {self.__dtype}.")
            x = x.astype(self.__dtype)

        m = self.__model
        if self.__normalize:
            z_samples = m.predict_f_samples(
                normalize_data(x, self.__X_normalizer),
                nsamples)  # shape = [S, N, D]

            # Transpose to shape [N, D, S]
            z_samples = __gpflow_format_to_gpy_format(z_samples)

            for i in range(z_samples.shape[0]):
                z_samples[i, :, :] = denormalize_data(
                    z_samples[i, :, :], self.__Y_normalizer)
        else:
            z_samples = m.predict_f_samples(
                x,
                nsamples)  # shape = [S, N, D]

            # Transpose to shape [N, D, S]
            z_samples = __gpflow_format_to_gpy_format(z_samples)

        return z_samples

    @property
    def data(self):
        return self.__data

    @property
    def model(self):
        return self.__model

    @property
    def kernel(self):
        return self.__kernel

    @property
    def n_features(self):
        return self.__n_features

    def __randomize(self, model: GPR):
        """
        Random initialization of trainable variables.
        """
        for param in model.trainable_parameters:
            random_value = tf.random.uniform(param.shape, dtype=self.__dtype)
            param.assign(random_value)

    def __initialize_likelihood_variance(self, model: GPR):
        """
        Random initialization of the noise variance.
        """
        random_value = tf.random.uniform(
            shape=[], dtype=self.__dtype, maxval=self.__noise_level)
        model.likelihood.variance.assign(
            random_value
            * tf.math.reduce_variance(model.data[1]))

    def __optimize_restart(self,
                           model: GPR,
                           run_optimizer: Callable[[GPR], GPR],
                           optimize_restarts: int,
                           verbose: bool = False,
                           plot_history: bool = True):
        """
        Perform specified number (optimize_restarts) of optimizations with
        randomized initial value to find the best solution.
        The GPR model which realizes the best likelihood is returned.
        """
        X, Y = self.__data
        best_likelihood = model.log_marginal_likelihood()
        best_model = copy.deepcopy(model)
        best_likelihood = model.log_marginal_likelihood()

        if plot_history:
            likelihood_history = []

        for i in range(optimize_restarts):
            model = gpflow.models.GPR(data=(X, Y),
                                      kernel=self.__design_kernel())

            # Random initialization.
            if hasattr(model.kernel, "initialize_hyperparameters"):
                # Random initialization of the kernel hyperparameters
                model.kernel.initialize_hyperparameters(X, Y)
                # Random initialization of the noise variance
                self.__initialize_likelihood_variance(model)
            else:
                # Random initialization of all trainables
                self.__randomize(model)

            # Optimization with the random initial parameters
            run_optimizer(model)

            likelihood = model.log_marginal_likelihood()

            if verbose:
                print(
                    f"Optimization restart {i + 1}/{optimize_restarts}, ",
                    "f = ", model.training_loss().numpy())
                gpflow.utilities.print_summary(model, "notebook")

            if plot_history:
                likelihood_history.append(likelihood.numpy().squeeze())

            # Compare the current likelihood with the best one ever known
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_model = copy.deepcopy(model)

        if plot_history:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(likelihood_history)
            ax.set_xlabel("Number of iterations")
            ax.set_ylabel("loglikelihood")
        return best_model
