import jax
import torch
import torchax
from torchax.interop import jax_view, jax_jit
torchax.enable_globally()



env = torchax.default_env()


def loss_fn():
    # env.manual_seed(key) 
    a =torch.randn((1, 2))
    jax.debug.print("{a}", a = a)
    b =torch.randn((1, 2))
    jax.debug.print("{b}", b = b)



new_loss_fn = jax_jit(loss_fn)


new_loss_fn()
new_loss_fn()


