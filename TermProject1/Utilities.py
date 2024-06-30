
from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class State(Enum):
    FREE = 0
    WALL = 1
    BOX = 2
    GOAL = 3
    PLAYER = 4
    BOX_ON_GOAL = 5  # Only for "stay" mode
    PLAYER_ON_GOAL = 6

