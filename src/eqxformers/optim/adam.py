import functools
import warnings

import optax

from ..optimizer_utils import make_weight_decay_mask, OptimizerConfig


@OptimizerConfig.register_subclass("adam")
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None  = 1.0


    def __post_init__(self):
        pass

    def make(self, num_train_steps):
        def _optimizer(learning_rate) -> optax.GradientTransformationExtraArgs:
            components = []
            if self.max_grad_norm:
               components.append(optax.clip_by_global_norm(self.max_grad_norm)) 

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)) 

            if self.weight_decay > 0:
                weight_decay = self.weight_decay
                if self.weight_decay_mask_pattern is None:
                    warnings.warn(
                        "Using weight decay without a mask pattern will apply weight decay to all parameters if this is not intended, please set weight_decay_mask_pattern accordingly."
                    )
                components.append(
                    optax.add_decayed_weights(
                        weight_decay,
                        functools.partial(make_weight_decay_mask, self.weight_decay_mask_pattern)
                    )
                ) 

            components.append(optax.scale(-learning_rate))
            tx = optax.chain(*components)
            return tx

        return optax.inject_hyperparams(_optimizer)(learning_rate = self.lr_scheduler(num_train_steps))


