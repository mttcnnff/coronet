from abc import ABC, abstractmethod
from typing import NamedTuple, Generator

import numpy as np

from .tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator(ABC):
    @abstractmethod
    def __call__(self, inputs: Tensor, targets: Tensor) -> Generator[Batch]:
        pass


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True,) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Generator[Batch]:
        assert len(inputs) == len(targets)

        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batched_inputs = inputs[start:end]
            batched_targets = targets[start:end]
            yield Batch(batched_inputs, batched_targets)
