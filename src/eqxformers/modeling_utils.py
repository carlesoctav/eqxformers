import typing as tp
import equinox as eqx
import copy


M = tp.TypeVar("M")

class PrepareFn(tp.Protocol):
    def __call__(self, *args: tp.Any) -> tuple:
        ...


no_hook = object()


#moduel with hooks
class Module(eqx.Module):
    prepare_input: PrepareFn = eqx.field(static=True, default=no_hook)
    prepare_output: PrepareFn = eqx.field(static=True, default=no_hook)

    def maybe_prepare_input(self, *args):
        if self.prepare_input is not no_hook:
            prepared = self.prepare_input(*args)
        else:
            prepared = args

        if isinstance(prepared, tuple):
            if len(prepared) == 1:
                return prepared[0]
            return prepared

        return prepared

    def maybe_prepare_output(self, *out):
        if self.prepare_output is not no_hook:
            prepared = self.prepare_output(*out)
        else:
            prepared = out

        if isinstance(prepared, tuple):
            if len(prepared) == 1:
                return prepared[0]
            return prepared

        return prepared



def add_prepare_fn(
        module: M,
        prepare_input: PrepareFn | None,
        prepare_output: PrepareFn | None,
):

    new_shell = copy.copy(module)
    if prepare_input:
        object.__setattr__(new_shell, "prepare_input", prepare_input)

    if prepare_output:
        object.__setattr__(new_shell, "prepare_output", prepare_output)

    return new_shell

