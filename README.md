# branch_extended_AD

Branch-extended automatic differentiation for fixed control-flow paths.

Use the existing JAX backend:

```python
import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp
```

Or use the optional PyTorch backend:

```python
from branch_extended_AD import pytorch as bead
import branch_extended_AD.torch_numpy as bnp
```

Both backends provide `record`, `replay`, `grad`, `value_and_grad`,
`replay_grad`, `replay_value_and_grad`, `replay_value_and_grad_batch`, and
`all_value_and_grad`.

Install only the backend you need:

```console
pip install 'branch_extended_AD[jax]'
pip install 'branch_extended_AD[torch]'
```

The PyTorch backend currently supports local absolute and relative tolerances.
Input-scaled tolerance remains available only with JAX. PyTorch batch replay is
implemented as a loop over paths; the JAX backend additionally provides JIT
compiled vectorized replay.
