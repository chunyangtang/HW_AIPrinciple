import random
from Utilities import Direction, State

DEFAULT_OBSTACLE_DENSITY = 0.1

MAP_REPRESENTATION = {
    State.FREE: " ",
    State.WALL: "#",
    State.BOX: "$",
    State.GOAL: ".",
    State.PLAYER: "@",
    State.BOX_ON_GOAL: "*",
    State.PLAYER_ON_GOAL: "+"
}


class SokobanMap:
    def __init__(self, width: int, height: int, num_boxes: int, num_goals: int = None, num_obstacles: int = None):
        self.width = width
        self.height = height
        assert width >= 3 and height >= 3, "Map must be at least 3x3."

        self.num_boxes = num_boxes
        self.num_goals = num_goals if num_goals is not None else num_boxes
        assert self.num_boxes <= self.num_goals, "Number of boxes must be <= goals, otherwise not solvable."
        self.num_obstacles = num_obstacles if num_obstacles is not None \
            else int((width - 1) * (height - 1) * DEFAULT_OBSTACLE_DENSITY)
        assert self.num_obstacles + self.num_goals + self.num_boxes < (width - 1) * (height - 1), \
            "Too many obstacles, goals, and boxes for the map size."

        self.map = [[State.FREE for _ in range(width)] for _ in range(height)]

        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.goal_positions = []  # not changed after generation
        self.box_positions = []
        self.player_x = self.player_y = None
        self.mode = "stay"  # "eat" or "stay" (eat = boxes moved to goals disappear, stay = boxes stay on goals)
        self.mode2 = "1toN"  # "1to1" or "1toN" (1to1 = one box to one goal, 1toN = one box to multiple goals)
        if self.mode2 == "1to1":
            assert self.num_boxes == self.num_goals, "1to1 mode requires equal number of boxes and goals."

    def is_inside_map(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        return self.map[y][x] == State.FREE

    def random_place(self, state: State, num: int, record_list=None):
        for _ in range(num):
            while True:  # Keep trying until we find a free spot
                x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)  # Avoid edge walls
                if self.is_free(x, y):
                    self.map[y][x] = state
                    if record_list is not None:
                        record_list.append((x, y))
                    break

    def place_obstacles(self):
        # Place around the edges
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.map[y][x] = State.WALL

        # Place random obstacles
        self.random_place(State.WALL, self.num_obstacles)

    def place_goals(self):
        # Place random goals
        self.random_place(State.GOAL, self.num_goals, self.goal_positions)

    def place_boxes(self):
        # Place random boxes
        self.random_place(State.BOX, self.num_boxes, self.box_positions)

    def place_player(self):
        while True:  # place the player at a random free spot
            x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            if self.is_free(x, y):
                self.map[y][x] = State.PLAYER
                self.player_x, self.player_y = x, y
                break

    def regenerate_map(self):
        self.map = [[State.FREE for _ in range(self.width)] for _ in range(self.height)]
        self.goal_positions = []
        self.box_positions = []
        self.place_obstacles()
        self.place_goals()
        self.place_boxes()
        self.place_player()

        return self.map

    def set_seed(self, seed):
        random.seed(seed)

    def get_map(self):
        return self.map

    def get_box_positions(self):
        return self.box_positions

    def get_goal_positions(self):
        return self.goal_positions

    def get_map_string(self):
        return "\n".join("".join(MAP_REPRESENTATION[state] for state in row) for row in self.map)

    def print_map(self):
        for row in self.map:
            print("".join(MAP_REPRESENTATION[state] for state in row))

    def make_move(self, direction: Direction):
        dx, dy = self.directions[direction.value]
        new_x, new_y = self.player_x + dx, self.player_y + dy
        if not self.is_inside_map(new_x, new_y):
            return False
        if self.map[new_y][new_x] == State.WALL:
            return False
        if self.map[new_y][new_x] in [State.BOX, State.BOX_ON_GOAL]:  # Try to move the box
            new_box_x, new_box_y = new_x + dx, new_y + dy
            if (not self.is_inside_map(new_box_x, new_box_y)
                    or self.map[new_box_y][new_box_x] not in [State.FREE, State.GOAL]):
                return False
            # Free the old box position
            self.map[new_y][new_x] = State.FREE if self.map[new_y][new_x] != State.BOX_ON_GOAL else State.GOAL
            box_index = self.box_positions.index((new_x, new_y))  # Find the box in the list, for 1to1 mode
            del self.box_positions[box_index]
            # Place the box to the new position
            if self.map[new_box_y][new_box_x] != State.GOAL:
                self.map[new_box_y][new_box_x] = State.BOX
                self.box_positions.append((new_box_x, new_box_y))
                if self.mode2 == "1to1":
                    goal = self.goal_positions.pop(box_index)
                    self.goal_positions.append(goal)  # Keep goal and box positions in sync

            else:  # Box moved to a goal
                if self.mode == "eat" and self.mode2 != "1to1":  # eat, 1toN
                    self.map[new_box_y][new_box_x] = State.GOAL  # Box disappears
                elif self.mode != "eat" and self.mode2 != "1to1":  # stay, 1toN
                    self.map[new_box_y][new_box_x] = State.BOX_ON_GOAL
                    self.box_positions.append((new_box_x, new_box_y))
                elif self.mode == "eat" and self.mode2 == "1to1":  # eat, 1to1
                    if (new_box_x, new_box_y) == self.goal_positions[box_index]:
                        self.map[new_box_y][new_box_x] = State.FREE  # Box and goal disappear simultaneously
                        del self.goal_positions[box_index]
                    else:
                        self.map[new_box_y][new_box_x] = State.BOX_ON_GOAL
                        goal = self.goal_positions.pop(box_index)
                        self.box_positions.append((new_box_x, new_box_y))
                        self.goal_positions.append(goal)  # Keep goal and box positions in sync
                else:  # stay, 1to1
                    self.map[new_box_y][new_box_x] = State.BOX_ON_GOAL
                    goal = self.goal_positions.pop(box_index)
                    self.box_positions.append((new_box_x, new_box_y))
                    self.goal_positions.append(goal)  # Keep goal and box positions in sync

        # Place the player
        self.map[self.player_y][self.player_x] = State.FREE if (self.map[self.player_y][self.player_x]
                                                                != State.PLAYER_ON_GOAL) else State.GOAL
        self.map[new_y][new_x] = State.PLAYER if (self.map[new_y][new_x] != State.GOAL) else State.PLAYER_ON_GOAL
        self.player_x, self.player_y = new_x, new_y
        return True

    def is_solved(self):  # i.e. all boxes are on goals (not necessarily all goals have boxes)
        if self.mode2 == "1to1":
            return self.box_positions == self.goal_positions
        if self.mode == "eat":
            return len(self.box_positions) == 0  # No boxes left
        else:  # mode == "stay"
            for box_x, box_y in self.box_positions:
                if self.map[box_y][box_x] != State.BOX_ON_GOAL:
                    return False
            return True

    def heuristic(self):
        if self.mode2 == "1to1":
            return sum(abs(box_x - goal_x) + abs(box_y - goal_y)
                       for (box_x, box_y), (goal_x, goal_y) in zip(self.box_positions, self.goal_positions))
        return sum(min(abs(box_x - goal_x) + abs(box_y - goal_y) for goal_x, goal_y in self.goal_positions)
                   for box_x, box_y in self.box_positions)


def main(stdscr):
    # Example usage
    smap = SokobanMap(10, 10, 3, 4)  # 10x10 map with 3 boxes
    smap.set_seed(13)  # solvable situation
    smap.mode = "eat"
    smap.mode2 = "1to1"
    smap.regenerate_map()

    stdscr.clear()

    stdscr.addstr("Use arrow keys to move, 'q' to quit.\n")
    stdscr.addstr(f"{smap.get_map_string()}\n")
    stdscr.addstr(f"Boxes: {str(smap.get_box_positions())}\n")
    stdscr.addstr(f"Goals: {str(smap.get_goal_positions())}\n")
    stdscr.addstr(f"Heuristic Score: {str(smap.heuristic())}")

    stdscr.refresh()
    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_UP:
            smap.make_move(Direction.UP)
        elif key == curses.KEY_DOWN:
            smap.make_move(Direction.DOWN)
        elif key == curses.KEY_LEFT:
            smap.make_move(Direction.LEFT)
        elif key == curses.KEY_RIGHT:
            smap.make_move(Direction.RIGHT)
        stdscr.clear()
        stdscr.addstr("Use arrow keys to move, 'q' to quit.\n")
        stdscr.addstr(f"{smap.get_map_string()}\n")
        stdscr.addstr(f"Boxes: {str(smap.get_box_positions())}\n")
        stdscr.addstr(f"Goals: {str(smap.get_goal_positions())}\n")
        stdscr.addstr(f"Heuristic Score: {str(smap.heuristic())}")
        stdscr.refresh()
        if smap.is_solved():
            stdscr.addstr("Solved!")
            stdscr.refresh()
            break


if __name__ == "__main__":
    import curses
    # For terminal use, arrow-keys to control and 'q' to quit
    curses.wrapper(main)








