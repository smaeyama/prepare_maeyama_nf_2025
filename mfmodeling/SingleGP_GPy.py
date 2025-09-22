#!/usr/bin/env python
# coding: utf-8

import numpy as np
import GPy

class SingleGP():
    '''
    Gaussian process regression.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        data : list[2]
            List of training data of y=f(x), where data input x and output y are vectors.
            [data_input[nsample,ninput], data_output[nsample,noutput]]
        augmented_dim : Int
            Augmented dimesnsion used for create the NARGP kernel.
            It this parameter is set, NARGP kernel is created.
        '''
        self.__data = kwargs.get("data")   # (List) data

        # Check if SingleGP is called for NARGP.
        self.__used_for_nargp = kwargs.get("augmented_dim") is not None
        self.__augmented_dim = kwargs.get("augmented_dim")
        return

    def optimize(self, **kwargs):
        '''
        Parameters
        ----------
        optimize_restarts : Int
        max_iters : Int
        verbose : Bool
            Parameters in GPy.models.GPRegression.optimize_restarts().
        '''
        # Get parameters
        optimize_restarts = kwargs.get("optimize_restarts", 30)
        max_iters = kwargs.get("max_iters", 400)
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("optimize_restarts=",optimize_restarts,", max_iters=",max_iters)

        # Single-fidelity GP
        X, Y = self.__data
        nsample, ninput = X.shape
        nsample, noutput = Y.shape

        # Design kernel function
        if self.__used_for_nargp:
            # NARGP kernel
            ninput = X.shape[1] - self.__augmented_dim
            noutput = self.__augmented_dim
            k = GPy.kern.RBF(ninput, active_dims=np.arange(ninput), ARD=True) \
                * GPy.kern.RBF(noutput, active_dims=np.arange(ninput,ninput+noutput), ARD=True) \
                + GPy.kern.RBF(ninput, active_dims=np.arange(ninput), ARD=True)
        else:
            k = GPy.kern.RBF(ninput, ARD=True)

        m = GPy.models.GPRegression(X=X, Y=Y, kernel=k)
        # Initialization of hyper parameters
        m[".*Gaussian_noise"] = m.Y.var()*0.01
        m[".*Gaussian_noise"].fix()
        # Optimization of hyper parameters
        m.optimize(max_iters=max_iters)
        m[".*Gaussian_noise"].unfix()
        m[".*Gaussian_noise"].constrain_positive()
        m.optimize_restarts(optimize_restarts, optimizer="bfgs", max_iters=max_iters, verbose=verbose)
        self.__model = m.copy()
        self.__kernel = k.copy()
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
        m = self.__model
        mu0, var0 = m.predict(x)
        return mu0, var0


    def predict_f_samples(self, x, num_samples):
        """
        Parameters
        ----------
        x : Numpy.ndarray[..., ninput]
            The points at which to make a prediction.
        num_samples : Int
            Number of samples to generate.

        Returns
        -------
        z_samples : Numpy.ndarray[..., noutput, num_samples]
        """
        m = self.__model
        z_samples = m.posterior_samples_f(
            x,
            num_samples)
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
