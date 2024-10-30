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
    shape_from_dims,
)
from pymc.logprob.transforms import Transform
from pymc.model.core import _UnsetType, Model
from pymc.util import UNSET
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
        "This is the distribution function"
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


def determine_shape(
    size: Size | None = None,
    shape: Shape | None = None,
    dims: Dims | None = None,
    observed: ndarray | Variable | None = None,
):
    """Determine the shape of the random variable.

    Preference is given to size or shape. If not specified, we rely on dims and
    finally, observed, to determine the shape of the variable.
    """
    if size is None and shape is None:
        if dims is not None:
            model = Model.get_context()
            shape = shape_from_dims(dims, model)
        elif observed is not None:
            shape = tuple(observed.shape)
    return shape


Normal = NormalDistribution()
