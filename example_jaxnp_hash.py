#!/usr/bin/env python3
import jax.numpy as jnp
import jaxnp_hash as jnph
import jaxnp_hash.numpy as jnp_h


def branching_function(x, y):
    result1 = jnp_h.maximum(x, y)
    result2 = jnp_h.abs(result1)
    result3 = jnp_h.max(result2)
    return result3


def simple_function(x):
    return jnp_h.max(x)


def aux_function(x, y):
    result = jnp_h.maximum(x, y)
    s = jnp_h.sum(result)
    return s, {"max_elem": jnp_h.max(result)}


x = jnp.array([1.0, 0.05, 0.94])
y = jnp.array([1.05, -1.5, 1.01])

print("=== Functional API Example ===\n")

print("1. Record and replay:")
val, paths = jnph.record(branching_function, tol=0.1)(x, y)
print(f"   Recorded value: {val}")
print(f"   Number of paths: {len(paths)}")

for i, p in enumerate(paths):
    replayed_val = jnph.replay(branching_function, p)(x, y)
    print(f"   Path {i+1}: {replayed_val}")
    print(f"     {paths.format_path(p)}")
print()

print("2. grad (record + differentiate w.r.t. x):")
g, paths = jnph.grad(simple_function, tol=0.1)(x)
print(f"   Gradient (default path): {g}")
print(f"   Number of paths: {len(paths)}")
print()

print("3. value_and_grad:")
(val, g), paths = jnph.value_and_grad(simple_function, tol=0.1)(x)
print(f"   Value: {val}, Gradient: {g}")
print(f"   Number of paths: {len(paths)}")
print()

print("4. replay_value_and_grad for each path:")
for i, p in enumerate(paths):
    v, g = jnph.replay_value_and_grad(simple_function, p)(x)
    print(f"   Path {i+1}: value={v}, grad={g}")
print()

print("5. value_and_grad with has_aux:")
(val, g, aux), paths = jnph.value_and_grad(aux_function, argnums=0, tol=0.1, has_aux=True)(x, y)
print(f"   Value: {val}")
print(f"   Gradient: {g}")
print(f"   Aux: {aux}")
print(f"   Number of paths: {len(paths)}")
print()

print("6. replay_grad with has_aux for each path:")
for i, p in enumerate(paths):
    g, aux = jnph.replay_grad(aux_function, p, argnums=0, has_aux=True)(x, y)
    print(f"   Path {i+1}: grad={g}, aux={aux}")

print("\n7. all_value_and_grad (all paths in one call):")
results, paths = jnph.all_value_and_grad(simple_function, tol=0.1)(x)
print(f"   Number of paths: {len(paths)}")
for i, (v, g) in enumerate(results):
    print(f"   Path {i+1}: value={v}, grad={g}")
print()

print("8. all_value_and_grad with has_aux:")
results, paths = jnph.all_value_and_grad(aux_function, argnums=0, tol=0.1, has_aux=True)(x, y)
print(f"   Number of paths: {len(paths)}")
for i, (v, g, aux) in enumerate(results):
    print(f"   Path {i+1}: value={v}, grad={g}, aux={aux}")

print("\n=== Functional API Demo Complete! ===")
