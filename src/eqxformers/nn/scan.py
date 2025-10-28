import abc 
import equinox as eqx
from ..modeling_utils import Module
from ..jax_utils import is_array_like
import typing as tp


Layer = tp.TypeVar("Layer", bound = Module)

class AbstractSequentialModule(tp.Genertic[Layer], eqx.Module):
    layers: eqx.AbstractVar[tuple[Layer, ...] | None | Layer]
    use_scan: eqx.AbstractVar[bool] 

    @abc.abstractmethod
    def call_with_scan(self, hidden_states, *args, **kwargs):
        pass


    @abc.abstractmethod
    def call_with_loop(self, hidden_states, *args, **kwargs):
        pass



def slice_out(tree, i):

    def take(leaf):
        if is_array_like(leaf):
            return leaf[i]
        else:
            return leaf


    return jtu.tree_map(take, tree)

