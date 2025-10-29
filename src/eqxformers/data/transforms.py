import dataclasses as dc
import typing as tp

from grain import transforms as grain_transforms
from jaxtyping import Array


Batch = tp.Any


@dc.dataclass
class CollateToBatch(grain_transforms.Map):
    """Convert feature dictionaries into typed batch containers."""

    batch_class: type[Batch]

    def map(self, features: dict[str, Array]) -> Batch:
        kwargs = {
            field.name: features.get(field.name)
            for field in dc.fields(self.batch_class)
        }
        return self.batch_class(**kwargs)
