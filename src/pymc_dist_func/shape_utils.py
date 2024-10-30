from numpy import ndarray
from pymc import Model
from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    Size,
    shape_from_dims,
)
from pytensor.graph.basic import Variable


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
