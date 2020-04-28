# def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
#               check_finite=True, bounds=(-np.inf, np.inf), method=None,
#               jac=None, **kwargs):
    """
    Use non-linear least squares to fit a function, f, to data.
    Assumes ``ydata = f(xdata, *params) + eps``
    Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : array_like or object
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, but can actually be any object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    p0 : array_like, optional
        Initial guess for the parameters (length N).  If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    sigma : None or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:
            - A 1-d `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.
            - A 2-d `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.
              .. versionadded:: 0.19
        None (default) is equivalent of 1-d `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.
        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is
        taken to be the same for all parameters.) Use ``np.inf`` with an
        appropriate sign to disable bounds on all or some parameters.
        .. versionadded:: 0.17
    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization.  See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.
        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.
        .. versionadded:: 0.18
    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.
    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.
        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.
    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.
    RuntimeError
        if the least-squares minimization fails.
    OptimizeWarning
        if covariance of the parameters can not be estimated.
    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    scipy.stats.linregress : Calculate a linear least squares regression for
                             two sets of measurements.
    Notes
    -----
    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
    through `leastsq`. Note that this algorithm can only deal with
    unconstrained problems.
    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
    the docstring of `least_squares` for more information.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import curve_fit
    >>> def func(x, a, b, c):
    ...     return a * np.exp(-b * x) + c
    Define the data to be fit with some noise:
    >>> xdata = np.linspace(0, 4, 50)
    >>> y = func(xdata, 2.5, 1.3, 0.5)
    >>> np.random.seed(1729)
    >>> y_noise = 0.2 * np.random.normal(size=xdata.size)
    >>> ydata = y + y_noise
    >>> plt.plot(xdata, ydata, 'b-', label='data')
    Fit for the parameters a, b, c of the function `func`:
    >>> popt, pcov = curve_fit(func, xdata, ydata)
    >>> popt
    array([ 2.55423706,  1.35190947,  0.47450618])
    >>> plt.plot(xdata, func(xdata, *popt), 'r-',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    Constrain the optimization to the region of ``0 <= a <= 3``,
    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:
    >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    >>> popt
    array([ 2.43708906,  1.        ,  0.35015434])
    >>> plt.plot(xdata, func(xdata, *popt), 'g--',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend()
    >>> plt.show()
    """
    # if p0 is None:
    #     # determine number of parameters by inspecting the function
    #     from scipy._lib._util import getargspec_no_self as _getargspec
    #     args, varargs, varkw, defaults = _getargspec(f)
    #     if len(args) < 2:
    #         raise ValueError("Unable to determine number of fit parameters.")
    #     n = len(args) - 1
    # else:
    #     p0 = np.atleast_1d(p0)
    #     n = p0.size
    #
    # lb, ub = prepare_bounds(bounds, n)
    # if p0 is None:
    #     p0 = _initialize_feasible(lb, ub)
    #
    # bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    # if method is None:
    #     if bounded_problem:
    #         method = 'trf'
    #     else:
    #         method = 'lm'
    #
    # if method == 'lm' and bounded_problem:
    #     raise ValueError("Method 'lm' only works for unconstrained problems. "
    #                      "Use 'trf' or 'dogbox' instead.")
    #
    # # optimization may produce garbage for float32 inputs, cast them to float64
    #
    # # NaNs can not be handled
    # if check_finite:
    #     ydata = np.asarray_chkfinite(ydata, float)
    # else:
    #     ydata = np.asarray(ydata, float)
    #
    # if isinstance(xdata, (list, tuple, np.ndarray)):
    #     # `xdata` is passed straight to the user-defined `f`, so allow
    #     # non-array_like `xdata`.
    #     if check_finite:
    #         xdata = np.asarray_chkfinite(xdata, float)
    #     else:
    #         xdata = np.asarray(xdata, float)
    #
    # if ydata.size == 0:
    #     raise ValueError("`ydata` must not be empty!")
    #
    # # Determine type of sigma
    # if sigma is not None:
    #     sigma = np.asarray(sigma)
    #
    #     # if 1-d, sigma are errors, define transform = 1/sigma
    #     if sigma.shape == (ydata.size, ):
    #         transform = 1.0 / sigma
    #     # if 2-d, sigma is the covariance matrix,
    #     # define transform = L such that L L^T = C
    #     elif sigma.shape == (ydata.size, ydata.size):
    #         try:
    #             # scipy.linalg.cholesky requires lower=True to return L L^T = A
    #             transform = cholesky(sigma, lower=True)
    #         except LinAlgError:
    #             raise ValueError("`sigma` must be positive definite.")
    #     else:
    #         raise ValueError("`sigma` has incorrect shape.")
    # else:
    #     transform = None
    #
    # func = _wrap_func(f, xdata, ydata, transform)
    # if callable(jac):
    #     jac = _wrap_jac(jac, xdata, transform)
    # elif jac is None and method != 'lm':
    #     jac = '2-point'
    #
    # if 'args' in kwargs:
    #     # The specification for the model function `f` does not support
    #     # additional arguments. Refer to the `curve_fit` docstring for
    #     # acceptable call signatures of `f`.
    #     raise ValueError("'args' is not a supported keyword argument.")
    #
    # if method == 'lm':
    #     # Remove full_output from kwargs, otherwise we're passing it in twice.
    #     return_full = kwargs.pop('full_output', False)
    #     res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
    #     popt, pcov, infodict, errmsg, ier = res
    #     ysize = len(infodict['fvec'])
    #     cost = np.sum(infodict['fvec'] ** 2)
    #     if ier not in [1, 2, 3, 4]:
    #         raise RuntimeError("Optimal parameters not found: " + errmsg)
    # else:
    #     # Rename maxfev (leastsq) to max_nfev (least_squares), if specified.
    #     if 'max_nfev' not in kwargs:
    #         kwargs['max_nfev'] = kwargs.pop('maxfev', None)
    #
    #     res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
    #                         **kwargs)
    #
    #     if not res.success:
    #         raise RuntimeError("Optimal parameters not found: " + res.message)
    #
    #     ysize = len(res.fun)
    #     cost = 2 * res.cost  # res.cost is half sum of squares!
    #     popt = res.x
    #
    #     # Do Moore-Penrose inverse discarding zero singular values.
    #     _, s, VT = svd(res.jac, full_matrices=False)
    #     threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    #     s = s[s > threshold]
    #     VT = VT[:s.size]
    #     pcov = np.dot(VT.T / s**2, VT)
    #     return_full = False
    #
    # warn_cov = False
    # if pcov is None:
    #     # indeterminate covariance
    #     pcov = zeros((len(popt), len(popt)), dtype=float)
    #     pcov.fill(inf)
    #     warn_cov = True
    # elif not absolute_sigma:
    #     if ysize > p0.size:
    #         s_sq = cost / (ysize - p0.size)
    #         pcov = pcov * s_sq
    #     else:
    #         pcov.fill(inf)
    #         warn_cov = True
    #
    # if warn_cov:
    #     warnings.warn('Covariance of the parameters could not be estimated',
    #                   category=OptimizeWarning)
    #
    # if return_full:
    #     return popt, pcov, infodict, errmsg, ier
    # else:
    #     return popt, pcov