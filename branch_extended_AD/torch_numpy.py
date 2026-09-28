import torch

from ._path_data import _AbsChoice, _ElementwiseChoice, _ReductionChoice
from .pytorch import _append, _pop, _state, _tolerance


def _coerce_pair(left, right):
    if isinstance(left, torch.Tensor):
        right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    elif isinstance(right, torch.Tensor):
        left = torch.as_tensor(left, dtype=right.dtype, device=right.device)
    else:
        left = torch.as_tensor(left)
        right = torch.as_tensor(right)
    return torch.broadcast_tensors(left, right)


def max(value):
    """Return the maximum and record nearby choices when active."""
    flat = value.reshape(-1)
    state = _state.get()
    if state.mode == "record":
        index = int(torch.argmax(flat))
        selected = flat[index]
        nearby = torch.nonzero(selected - flat <= _tolerance(selected)).reshape(-1).tolist()
        _append("max", [_ReductionChoice(int(item)) for item in nearby])
    elif state.mode == "replay":
        index = _pop("max").index
    else:
        return torch.max(value)
    return flat[index]


def min(value):
    """Return the minimum and record nearby choices when active."""
    flat = value.reshape(-1)
    state = _state.get()
    if state.mode == "record":
        index = int(torch.argmin(flat))
        selected = flat[index]
        nearby = torch.nonzero(flat - selected <= _tolerance(selected)).reshape(-1).tolist()
        _append("min", [_ReductionChoice(int(item)) for item in nearby])
    elif state.mode == "replay":
        index = _pop("min").index
    else:
        return torch.min(value)
    return flat[index]


def _elementwise(left, right, name, operation):
    left, right = _coerce_pair(left, right)
    state = _state.get()
    if state.mode == "record":
        reference = torch.maximum(left.abs(), right.abs())
        nearby = torch.nonzero((left - right).abs() <= _tolerance(reference)).reshape(-1).tolist()
        base = (operation(left, right) == right).reshape(-1).tolist()
        choices = [
            _ElementwiseChoice(tuple(nearby), bits, tuple(base))
            for bits in range(2 ** len(nearby))
        ]
        _append(name, choices)
        bits = 0
    elif state.mode == "replay":
        choice = _pop(name)
        nearby, base, bits = choice.nearby_indices, choice.base_pick_two, choice.choice_bits
    else:
        return operation(left, right)
    if state.mode == "record":
        nearby, base = tuple(nearby), tuple(base)
    pick_right = torch.tensor(base, dtype=torch.bool, device=left.device)
    if nearby:
        indices = torch.tensor(nearby, dtype=torch.long, device=left.device)
        flips = ((bits >> torch.arange(len(nearby), device=left.device)) & 1).bool()
        pick_right[indices] = torch.logical_xor(pick_right[indices], flips)
    return torch.where(pick_right.reshape(left.shape), right, left)


def maximum(left, right):
    """Return elementwise maxima and record nearby choices when active."""
    return _elementwise(left, right, "maximum", torch.maximum)


def minimum(left, right):
    """Return elementwise minima and record nearby choices when active."""
    return _elementwise(left, right, "minimum", torch.minimum)


def sum(value):
    """Return the sum of all tensor entries."""
    return torch.sum(value)


def abs(value):
    """Return absolute values using the active branch policy."""
    state = _state.get()
    if state.mode == "inactive":
        return torch.abs(value)
    if state.mode == "record":
        nearby = tuple(torch.nonzero(value.abs().reshape(-1) <= _tolerance(value).reshape(-1)).reshape(-1).tolist())
        base = tuple((value < 0).reshape(-1).tolist())
        if state.abs_policy == "enumerate":
            choices = [_AbsChoice(nearby, base, bits) for bits in range(2 ** len(nearby))]
        else:
            choices = [_AbsChoice(nearby, base)]
        _append("abs", choices)
        choice = choices[0]
    else:
        choice = _pop("abs")
    negate = torch.tensor(choice.base_negate, dtype=torch.bool, device=value.device)
    if choice.choice_bits is not None and choice.nearby_indices:
        indices = torch.tensor(choice.nearby_indices, dtype=torch.long, device=value.device)
        flips = ((choice.choice_bits >> torch.arange(len(choice.nearby_indices), device=value.device)) & 1).bool()
        negate[indices] = torch.logical_xor(negate[indices], flips)
    linear = torch.where(negate.reshape(value.shape), -value, value)
    if choice.choice_bits is None and choice.nearby_indices:
        ambiguous = torch.zeros(value.numel(), dtype=torch.bool, device=value.device)
        ambiguous[list(choice.nearby_indices)] = True
        linear = torch.where(ambiguous.reshape(value.shape), value.abs().detach(), linear)
    return linear
