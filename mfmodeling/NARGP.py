#!/usr/bin/env python
# coding: utf-8
import copy
import numpy as np

from mfmodeling import use_gpy_in_nargp
from .utils import normalize_data_list_columnwise, \
    denormalize_mean, denormalize_variance, \
    normalize_data

class NARGP():
    '''
    Nonlinear autoregressive multi-fidelity Gaussian process regression (NARGP).
    P. Perdikaris, et al., "Nonlinear information fusion algorithms for data-efficient multi-fidelity modeling"
    Proc. R. Soc. A 473, 20160751 (2017). http://dx.doi.org/10.1098/rspa.2016.0751
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        data_list : list[nfidelity]
            List of multi-fidelity training data of y=f(x), where input x and output y are vectors.

            Structure of the list of data is as follow.
            data_list[ 0] = [data_input_lowest[nsample_lowest,ninput],   data_output_lowest[nsample_lowest,noutput_lowest]]
            data_list[ 1] = [data_input_1st[nsample_1st,ninput],         data_output_1st[nsample_1st,noutput_1st]]
            ...
            data_list[-1] = [data_input_highest[nsample_highest,ninput], data_output_highest[nsample_highest,noutput_highest]]

            "ninput" (dimension of the input vector x) is the same for all fidelities, while "noutput_***" (dimension of
            the output vector y) and "nsample_***" (number of sampling of data set) can be different for each fidelity.
        single_gp_config_list : list[dict]
            List of dictionaries containing the arguments for each SingleGP class constructor.
            single_gp_config_list[i] contains the arguments for i-th fidelity data.
        normalize : Bool
            Options for normalizing input data list.
        '''
        # Select package for SingleGP class
        if use_gpy_in_nargp():
            from .SingleGP_GPy import SingleGP
            print("NARGP is using GPy.")
        else:
            from .SingleGP_GPflow import SingleGP
            print("NARGP is using GPflow.")
        self.SingleGP = SingleGP

        # Check key word arguments
        for kwarg in kwargs:
            if kwarg not in {"data_list",
                             "single_gp_config_list",
                             "normalize"}:
                raise TypeError(f"Unknown keyword argument for NARGP: {kwarg}")

        # (List) training data
        if kwargs.get("normalize") is None:
            self.__normalize = False
        else:
            self.__normalize = kwargs.get("normalize")

        # Set normalizer by all fidelity data
        data_list = kwargs.get("data_list")
        self.__X_normalizer_list = []
        self.__Y_normalizer_list = []
        for data in data_list:
            self.__X_normalizer_list.append(data[0])
            self.__Y_normalizer_list.append(data[1])

        # (List) data_list[nfidelity]
        self.__data_list = []
        if self.__normalize:
            # X is normalized using all fidelity data
            X_list_normalized = normalize_data_list_columnwise(
                self.__X_normalizer_list, self.__X_normalizer_list
            )

            # Y is normalized fidelity-wise
            Y_list_normalized = []
            for Y in self.__Y_normalizer_list:
                Y_normalized = normalize_data(Y, Y)
                Y_list_normalized.append(Y_normalized)

            for X, Y in zip(X_list_normalized, Y_list_normalized):
                self.__data_list.append([X, Y])
        else:
            self.__data_list = data_list

        # Arguments for instantiating the SingleGP class at each fidelity
        default_single_gp_config_list = [{
            "kernel": "RBF",
            "kernel_args": {}
        }] * len(self.__data_list)

        single_gp_config_list = kwargs.get(
            "single_gp_config_list",
            default_single_gp_config_list)

        # If config length does not match data length, throw ValueError
        if len(single_gp_config_list) != len(self.__data_list):
            raise ValueError("NARGP: SingleGP config list length does not mach data length.")

        # (List) single_gp_config_list[nfidelity]
        self.__single_gp_config_list = single_gp_config_list

        # (List) single_gp_list[nfidelity]
        self.__single_gp_list = []


    def optimize(self, **kwargs):
        '''
        Optimization of the hyperparameters in NARGP.

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
        nMonteCarlo : Int
            Number of Monte Carlo sampling points for prediction called in NARGP.
        verbose : Bool
            Shows progress of the __optimize_restart if True.
        plot_history : Bool
            Shows history of the objective function in the __optimize_restart if True.
        optimize_noise_in_first_step : Bool
            Optimize noise level in the first step if True.
        use_built_in_initialization : Bool
            Use hyper parameter initialization method (if any) of the kernel.
        '''
        # Check key word arguments
        for kwarg in kwargs:
            if kwarg not in {
                "optimize_restarts",
                "max_iters",
                "optimizer_name",
                "optimizer_parameters",
                "nMonteCarlo",
                "verbose",
                "plot_history",
                "optimize_noise_in_first_step",
                "use_built_in_initialization"
                }:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        # Get variables from key word arguments
        verbose = kwargs.get("verbose", False)

        optimize_restarts = kwargs.get("optimize_restarts", 30)

        max_iters = kwargs.get("max_iters", 400)

        nfidelity = len(self.__data_list)

        nMonteCarlo = kwargs.pop("nMonteCarlo", 1000)

        if verbose:
            print("nfidelity=", nfidelity,
                  ", optimize_restarts=", optimize_restarts,
                  ", max_iters=", max_iters)

        # Initialization of SingleGP object list
        self.__single_gp_list = []

        # Single-fidelity GP for the lowest fidelity data
        ifidelity = 0
        X, Y = self.__data_list[ifidelity]

        # SingleGP class object for the lowest fidelity data
        single_gp = self.SingleGP(
            data=(X, Y),
            kernel=self.__single_gp_config_list[ifidelity].get("kernel"),
            kernel_args=self.__single_gp_config_list[ifidelity].get("kernel_args"),
            normalize=False)

        # Optimization of hyperparameters
        # Arguments are passed to SingleGP.optimize
        single_gp.optimize(**kwargs)

        self.__single_gp_list.append(copy.deepcopy(single_gp))

        # Multi-fidelity GP for higher fidelity data
        if nfidelity > 1:
            for ifidelity in range(1, nfidelity):
                X, Y = self.__data_list[ifidelity]

                mu, _ = self.predict(X, ifidelity=ifidelity-1,
                                     nMonteCarlo=nMonteCarlo,
                                     called_in_optimize=True)

                # Augmented data in NARGP
                XX = np.hstack((X, mu))

                # SingleGP class object for the ifidelity-th fidelity data
                single_gp = self.SingleGP(
                    data=(XX, Y),
                    kernel=self.__single_gp_config_list[ifidelity].get("kernel"),
                    kernel_args=self.__single_gp_config_list[ifidelity].get("kernel_args"),
                    augmented_dim=mu.shape[1],
                    normalize=False)

                # Optimization of hyperparameters
                single_gp.optimize(**kwargs)

                self.__single_gp_list.append(copy.deepcopy(single_gp))
        return

    def __generate_montecarlo_samples(
            self, level, x, z_samples, ifidelity, nMonteCarlo):
        """
        Recursive generation of the Monte Carlo samples for prediction.
        Due to the large overhead of calling predict method for nMonteCarlo times,
        we call the predict once with the combined data.
        """
        # List of mean and variance
        mu_list = []
        var_list = []

        if level == ifidelity - 1:
            # Last recursion of the recursive function
            data_set = [
                np.hstack([x, z_samples[:, :, i]])
                for i in range(nMonteCarlo)
            ]

            concatenated_data = np.concatenate(data_set, axis=0)

            mu_all, var_all = self.__single_gp_list[ifidelity].predict(
                concatenated_data)

            split_indices = np.cumsum([ds.shape[0] for ds in data_set])[:-1]
            split_mus = np.split(mu_all, split_indices)
            split_vars = np.split(var_all, split_indices)

            mu_list = [mu for mu in split_mus]
            var_list = [var for var in split_vars]
            return mu_list, var_list
        else:
            for i in range(nMonteCarlo):
                # Recursive generation of the Monte Carlo samples
                z_samples_next = self.__single_gp_list[level + 1].predict_f_samples(
                    np.hstack((x, z_samples[:, :, i])), nMonteCarlo)

                mu_list_next, var_list_next = self.__generate_montecarlo_samples(
                    level + 1, x, z_samples_next, ifidelity, nMonteCarlo)

                mu_list.extend(mu_list_next)
                var_list.extend(var_list_next)
            return mu_list, var_list

    def predict(self,
                x,
                ifidelity=None,
                nMonteCarlo=1000,
                called_in_optimize=False):
        '''
        Parameters
        ----------
        x : Numpy.ndarray[..., ninput]
            The points at which to make a prediction
        ifidelity : Int
            0 < ifidelity < nfidelity-1
            Prediction for the specified fidelity model
        nMonteCarlo : Int
            Sampling number of Monte Carlo integration of Eq. (2.14)
        called_in_optimize : Bool


        Returns
        -------
        mean : Numpy.ndarray[..., noutput]
            Mean for i-fidelity model
        variance : Numpy.ndarray[..., noutput]
            Variance for i-th fidelity model
        '''
        if ifidelity is None:
            ifidelity = len(self.__data_list) - 1

        # If not called in optimize method,
        # perform normalization if required
        if called_in_optimize:
            # If called in optimize method, skip normalization check
            x_new = x
        else:
            if self.__normalize:
                x_normalized = normalize_data_list_columnwise(
                    [x], self.__X_normalizer_list)
                x_new = x_normalized[0]
            else:
                x_new = x

        if ifidelity == 0: # Evaluate at fidelity level 0
            mu, var = self.__single_gp_list[0].predict(x_new)
        else:
            z_samples = self.__single_gp_list[0].predict_f_samples(x_new, nMonteCarlo)

            # Generate Monte Carlo samples
            mu_list, var_list = self.__generate_montecarlo_samples(
                level=0,
                x=x_new,
                z_samples=z_samples,
                ifidelity=ifidelity,
                nMonteCarlo=nMonteCarlo)

            mu = np.mean(mu_list, axis=0)
            var = np.mean(var_list, axis=0) + np.var(mu_list, axis=0)
            var = np.abs(var)

        # Denormalize if required
        if called_in_optimize:
            return mu, var
        else:
            if self.__normalize:
                mu = denormalize_mean(
                    mu, self.__Y_normalizer_list[ifidelity])
                var = denormalize_variance(
                    var, self.__Y_normalizer_list[ifidelity])
                return mu, var
            else:
                return mu, var

    @property
    def data_list(self):
        return self.__data_list

    @property
    def single_gp_list(self):
        return self.__single_gp_list

    @property
    def model_list(self):
        return [single_gp.model for single_gp in self.single_gp_list]

    @property
    def kernel_list(self):
        return [single_gp.kernel for single_gp in self.single_gp_list]
