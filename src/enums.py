from __future__ import annotations

from enum import Enum, auto


class EdgeKind(Enum):
    PRIMARY = auto()
    CROSSLINK = auto()


class PointerIntent(Enum):
    NONE = auto()
    SEED = auto()
    TAP = auto()
    LONG_PRESS = auto()
    DRAG = auto()
    FLICK = auto()
    SWIRL = auto()
