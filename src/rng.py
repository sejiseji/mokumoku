from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RandomSource:
    seed: int
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def uniform(self, low: float, high: float) -> float:
        return self._random.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def choice_index(self, count: int) -> int:
        if count <= 0:
            raise ValueError("count must be positive")
        return self._random.randrange(count)
