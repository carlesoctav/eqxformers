import equinox as eqx
import jax
from src.eqxformers import nn



@eqx.filter_vmap
def make_module(key):
    return nn.Linear(10, 10, key = key)



key = jax.random.key(10)
module = make_module(jax.random.split(key, 3))
inputs = jax.random.normal(key, (1, 10))

@eqx.filter_jit
def rollout(inputs, module, keys):

    dy, st = eqx.partition(module, eqx.is_array)
    print(f"DEBUGPRINT[88]: test_scan.py:18: keys={keys}")
    def do_scan(carry, xs):
        dy, key = xs
        layer = eqx.combine(dy, st) 
        carry = layer(carry)
        return carry, None

    carry, _ =  jax.lax.scan(do_scan, inputs, xs = (dy, keys))
    return carry



hlo = rollout.lower(inputs, module, jax.random.split(key, 3)).as_text()

with open("./simple_roll.txt",  "w") as f:
    f.write(hlo)

