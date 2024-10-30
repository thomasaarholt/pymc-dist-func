# Use the pymc master branch
from collections.abc import Callable
from typing import Literal

from numpy._typing import ArrayLike
import pymc as pm
from numpy import ndarray
from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    Size,
    convert_shape,
    convert_size,
    find_size,
)
from pymc.logprob.transforms import Transform
from pymc.model.core import _UnsetType
from pymc.util import UNSET
from pymc_dist_func.shape_utils import determine_shape
from pytensor.graph.basic import Variable
from pytensor.graph.utils import MetaType
from pytensor.tensor.variable import TensorVariable
from pytensor.tensor.random.basic import NormalRV


def add_dist(
    rv_out: TensorVariable,
    name: str,
    dims: Dims | None = None,
    initval: Literal["support_point", "prior"] | float | None = None,
    observed: ndarray | Variable | None = None,
    total_size: float | None = None,
    transform: Transform | _UnsetType = UNSET,
    default_transform: Transform | _UnsetType = UNSET,
):
    """Add any rv to the model"""
    if observed is not None:
        observed = pm.convert_observed_data(observed)
    model = pm.modelcontext(None)
    rv_out = model.register_rv(
        rv_out,
        name,
        observed=observed,
        total_size=total_size,
        dims=dims,
        transform=transform,
        default_transform=default_transform,
        initval=initval,
    )
    return rv_out


def dist(
    *dist_params: ArrayLike | TensorVariable,
    rv_op: Callable[..., TensorVariable],
    rv_type: MetaType | None = None,
    size: Size | None = None,
    shape: Shape | None = None,
) -> TensorVariable:
    ndim_supp = getattr(rv_op, "ndim_supp", getattr(rv_type, "ndim_supp", None))
    if ndim_supp is None:
        # Initialize Ops and check the ndim_supp that is now required to exist
        ndim_supp = rv_op(*dist_params, size=size, shape=shape).owner.op.ndim_supp

    shape = convert_shape(shape)
    size = convert_size(size)
    create_size = find_size(shape=shape, size=size, ndim_supp=ndim_supp)
    rv_out = rv_op(*dist_params, size=create_size)
    return rv_out


class NormalDistribution:
    @staticmethod
    def dist(
        mu: ArrayLike | TensorVariable,
        sigma: ArrayLike | TensorVariable,
        dims: Dims | None = None,
        size: Size | None = None,
        shape: Shape | None = None,
    ) -> TensorVariable:
        """This is the normal.dist method"""
        normal_rv = NormalRV()
        return dist(mu, sigma, rv_op=normal_rv, shape=shape)

    def __call__(
        self,
        name: str,
        mu: float,
        sigma: float,
        dims: Dims | None = None,
        initval: Literal["support_point", "prior"] | float | None = None,
        size: Size | None = None,
        shape: Shape | None = None,
        observed: ndarray | Variable | None = None,
        total_size: float | None = None,
        transform: Transform | _UnsetType = UNSET,
        default_transform: Transform | _UnsetType = UNSET,
    ):
        r"""
        Univariate normal log-likelihood.

        The pdf of this distribution is

        .. math::

           f(x \mid \mu, \tau) =
               \sqrt{\frac{\tau}{2\pi}}
               \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

        Normal distribution can be parameterized either in terms of precision
        or standard deviation. The link between the two parametrizations is
        given by

        .. math::

           \tau = \dfrac{1}{\sigma^2}

        .. plot::
            :context: close-figs

            import matplotlib.pyplot as plt
            import numpy as np
            import scipy.stats as st
            import arviz as az
            plt.style.use('arviz-darkgrid')
            x = np.linspace(-5, 5, 1000)
            mus = [0., 0., 0., -2.]
            sigmas = [0.4, 1., 2., 0.4]
            for mu, sigma in zip(mus, sigmas):
                pdf = st.norm.pdf(x, mu, sigma)
                plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
            plt.xlabel('x', fontsize=12)
            plt.ylabel('f(x)', fontsize=12)
            plt.legend(loc=1)
            plt.show()

        ========  ==========================================
        Support   :math:`x \in \mathbb{R}`
        Mean      :math:`\mu`
        Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
        ========  ==========================================

        Parameters
        ----------
        mu : tensor_like of float, default 0
            Mean.
        sigma : tensor_like of float, optional
            Standard deviation (sigma > 0) (only required if tau is not specified).
            Defaults to 1 if neither sigma nor tau is specified.
        tau : tensor_like of float, optional
            Precision (tau > 0) (only required if sigma is not specified).

        Examples
        --------
        .. code-block:: python

            with pm.Model():
                x = pm.Normal("x", mu=0, sigma=10)

            with pm.Model():
                x = pm.Normal("x", mu=0, tau=1 / 23)
        """

        shape = determine_shape(size, shape, dims, observed)

        rv_out = self.dist(mu=mu, sigma=sigma, dims=dims, size=size, shape=shape)
        add_dist(
            rv_out=rv_out,
            name=name,
            dims=dims,
            initval=initval,
            observed=observed,
            total_size=total_size,
            transform=transform,
            default_transform=default_transform,
        )
        return rv_out


Normal = NormalDistribution()
