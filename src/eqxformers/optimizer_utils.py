import abc
import fnmatch
from dataclasses import dataclass

import draccus
import equinox as eqx
import jax.tree_util as jtu
import numpy as np
import optax
from jaxtyping import Array


@dataclass(frozen = True)
class OptimizerConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    #maybe make this abstract class var

    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup: int | float = 0.01
    lr_decay: int | float | None = None
    re_warmup: int| float = 0.0
    lr_schedule: str = "constant"

    cycle_length: int| float| None | list[int] = None
    cycles: int | list[int]  | None = None

    weight_decay: float = 0.1
    weight_decay_mask_pattern: list[str] | str | None = None

    @abc.abstractmethod
    def make(self, num_train_steps: int):
        raise NotImplementedError

    def _get_cycle_minima(self, total_main_steps):
        if self.cycle_length is not None and self.cycles is not None:
            raise ValueError("Only one of cycle_length or cycles can be set")

        if self.cycle_length:
            if isinstance(self.cycle_length, int | float):
                cycle_length = _convert_frac_or_steps(self.cycle_length, total_main_steps) 
                points = [i * cycle_length for i in range(1, total_main_steps // cycle_length)] 
            elif isinstance(self.cycle_length, list):
                steps = np.cumsum(np.asarray(self.cycle_length)) 
                if steps[-1]  > total_main_steps:
                    raise ValueError("Sum of cycle_length list exceeds total training steps")
                points = steps.tolist()
            else:
                raise ValueError("cycle_length must be int, float, or list of int")
        elif self.cycles:
            raise NotImplementedError("cycles not implemented yet")

        points.insert(0, 0)
        if points[-1] != total_main_steps:
            points.append(total_main_steps)

        return points



    def lr_scheduler(self, num_train_steps, override_lr = None):
        learning_rate = self.lr

        min_lr = learning_rate * self.min_lr_ratio
        points = self._get_cycle_minima(num_train_steps)
        schedules = []
        boundaries =[]
        previous_end = 0.0

        for cycle, (start, end) in enumerate(zip(points[:-1], points[1:])):
            cycle_steps = end - start
            warmup_setting = self.warmup if cycle == 0 else self.re_warmup
            warmup_steps = _convert_frac_or_steps(warmup_setting, cycle_steps)

            if warmup_steps != 0:
                warmup = optax.linear_schedule(previous_end, learning_rate, warmup_steps)
                schedules.append(warmup)
                boundaries.append(start + warmup_steps)

            lr_decay_steps = (
                _convert_frac_or_steps(self.lr_decay, cycle_steps)
                if self.lr_decay is not None else 
                cycle_steps - warmup_steps
            )

            stable_steps = cycle_steps  - warmup_steps - lr_decay_steps
            print(f"DEBUGPRINT[26]: optimizer_utils.py:88: stable_steps={stable_steps}")

            if stable_steps != 0:
                stable = optax.constant_schedule(learning_rate)
                schedules.append(stable)
                boundaries.append(start + warmup_steps + stable_steps)

            if isinstance(self.lr_schedule, str):
                match self.lr_schedule:
                    case "constant":
                        decay = optax.constant_schedule(learning_rate)
                        schedules.append(decay)
                        boundaries.append(end)
                    case "linear":
                        decay = optax.linear_schedule(learning_rate,min_lr, lr_decay_steps)
                        schedules.append(decay)
                        boundaries.append(end)
                    case "cosine":
                        decay = optax.cosine_decay_schedule(learning_rate, lr_decay_steps)
                        boundaries.append(end) 
                    case _:
                        raise NotImplementedError
                
            previous_end = decay(lr_decay_steps)

        print(f"DEBUGPRINT[25]: optimizer_utils.py:114: schedules={schedules}")
        print(f"DEBUGPRINT[24]: optimizer_utils.py:114: boundaries={boundaries}")
        if len(schedules) > 1:
            return optax.join_schedules(schedules, boundaries) 
        else:
            return schedules[0]


def _convert_frac_or_steps(
    length_or_frac: int | float,
    total_step: int,
)-> int:
    if isinstance(length_or_frac, float):
        if  0<= length_or_frac <=1:
            return int(length_or_frac * total_step)
        else:
            raise ValueError("If cycle_length is a float, it must be between 0 and 1") 
    else:
        return length_or_frac


def make_weight_decay_mask(patterns: list[str], module):
    def f(path, leaf):
        if isinstance(leaf, Array):
            if patterns is None:
                return True

            for pattern in paterns:
                if fnmatch.fnmatchcase(jtu.keystr(path), pattern):
                    return True
            return False
        else:
            return False

    return jtu.tree_map_with_path(f, module, is_leaf = eqx.is_array)
