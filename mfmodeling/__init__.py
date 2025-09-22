from .SingleGP_GPflow import SingleGP

# Use GPy in NARGP if True.
_use_gpy_in_nargp = False


def set_gpy_usage_in_nargp(use_gpy_in_nargp: bool):
    """
    Set GPy usage in NARGP class.

    Parameters
    ----------
    use_gpy : Bool
        A bool whether NARGP uses GPy.
    """
    global _use_gpy_in_nargp
    _use_gpy_in_nargp = use_gpy_in_nargp


def use_gpy_in_nargp():
    """
    Getter of GPy usage in NARGP class.

    Returns
    -------
    Bool
        A bool whether NARGP uses GPy.
    """
    global _use_gpy_in_nargp
    return _use_gpy_in_nargp

from .NARGP import NARGP