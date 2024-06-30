
import queue
from copy import deepcopy
from typing import List
from dataclasses import dataclass
from Utilities import Direction
from SokobanMap import SokobanMap

MAX_MOVE_RATE = 1    # Search will stop after a state exceed MAX_MOVE_RATE*SokobanMap.width*SokobanMap.height moves


@dataclass
class SokobanState:
    moves: int  # Trajectory cost, for search priority sorting
    map: SokobanMap  # Current state, for search
    trajectory: List[Direction]  # Trajectory to reach this state, for backtracking

    def __lt__(self, other):  # Shorter path get higher priority
        return self.moves > other.moves


class SokobanSearcher:
    def __init__(self, sokoban_map: SokobanMap):
        self._initial_map = sokoban_map
        self.max_moves = MAX_MOVE_RATE * sokoban_map.width * sokoban_map.height

    def astar(self):
        pq = queue.PriorityQueue()
        pq.put((0, SokobanState(0, deepcopy(self._initial_map), [])))
        visited = set()
        while True:
            if pq.empty():
                break
            _, current_state = pq.get()
            if current_state.map.get_map_string() in visited:
                continue
            visited.add(current_state.map.get_map_string())

            if current_state.map.is_solved():
                return current_state
            if current_state.moves > self.max_moves:
                break

            sokoban = current_state.map  # type: SokobanMap
            for direction in Direction:
                new_map = deepcopy(sokoban)
                if new_map.make_move(direction):
                    new_state = SokobanState(current_state.moves + 1, new_map, current_state.trajectory + [direction])
                    pq.put((new_state.moves + new_state.map.heuristic(), new_state))

        return None
