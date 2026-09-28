_JAX_EXPORTS = {
    "record",
    "replay",
    "grad",
    "value_and_grad",
    "replay_grad",
    "replay_value_and_grad",
    "replay_value_and_grad_batch",
    "all_value_and_grad",
}

from ._path_data import PathSet

__all__ = ["PathSet", *sorted(_JAX_EXPORTS)]


def __getattr__(name):
    if name not in _JAX_EXPORTS:
        raise AttributeError(name)
    try:
        from . import HashTensor
    except ModuleNotFoundError as error:
        if error.name in {"jax", "jaxlib", "numpy"}:
            raise ModuleNotFoundError(
                "The JAX backend is not installed. Install branch_extended_AD[jax] "
                "or import branch_extended_AD.pytorch."
            ) from error
        raise
    value = getattr(HashTensor, name)
    globals()[name] = value
    return value
