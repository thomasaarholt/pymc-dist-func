from functools import wraps
from typing import Callable, Literal, ParamSpec, TypeVar

import pymc as pm
from numpy import ndarray
from numpy._typing import ArrayLike
from pymc.distributions.shape_utils import Dims, Shape, Size
from pymc.logprob.transforms import Transform
from pymc.model.core import _UnsetType
from pymc.util import UNSET
from pytensor.graph.basic import Variable
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.variable import TensorVariable

from pymc_dist_func.distribution import dist
from pymc_dist_func.shape_utils import determine_shape

P = ParamSpec("P")
T = TypeVar("T")


def distribution_registration(
    func: Callable[P, TensorVariable],
) -> Callable[P, TensorVariable]:
    """Decorator that handles PyMC distribution registration"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TensorVariable:
        # Extract 'name' from args or kwargs
        name = args[1] if len(args) > 1 else kwargs.get("name")

        # Call the original function to get the rv_out
        rv_out = func(*args, **kwargs)

        # Extract the necessary parameters for add_dist
        dims = kwargs.get("dims")
        initval = kwargs.get("initval")
        observed = kwargs.get("observed")
        total_size = kwargs.get("total_size")
        transform = kwargs.get("transform", UNSET)
        default_transform = kwargs.get("default_transform", UNSET)

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

    return wrapper


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

    @distribution_registration
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
    ) -> TensorVariable:
        """Univariate normal log-likelihood."""
        shape = determine_shape(size, shape, dims, observed)
        rv_out = self.dist(mu=mu, sigma=sigma, dims=dims, size=size, shape=shape)
        return rv_out


Normal = NormalDistribution()
