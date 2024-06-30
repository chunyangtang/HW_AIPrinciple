
from SokobanMap import SokobanMap
from SokobanGUI import SokobanGUI
from SokobanSearcher import SokobanSearcher


def main1():  # Example usage of creating a SokobanMap and SokobanGUI
    # Create a SokobanMap with 20x20 size, 6 boxes, 8 goals, and 40 obstacles
    sokoban_map = SokobanMap(20, 20, 6, 8, 40)
    sokoban_map.regenerate_map()
    gui = SokobanGUI(sokoban_map.get_map())
    gui.draw_map()
    gui.mainloop()


def main2():  # Example usage of Problem 2
    sokoban_map = SokobanMap(10, 10, 3, 4)
    sokoban_map.set_seed(66)  # tested solvable situation
    sokoban_map.mode = "stay"
    sokoban_map.mode2 = "1toN"

    sokoban_map.regenerate_map()
    gui = SokobanGUI(sokoban_map.get_map())
    gui.draw_map()
    searcher = SokobanSearcher(sokoban_map)
    while (state := searcher.astar()) is None:
        print("No solution found")
        sokoban_map.print_map()
        sokoban_map.regenerate_map()
    for direction in state.trajectory:
        sokoban_map.make_move(direction)
        gui.window_after(sokoban_map.get_map())
    gui.mainloop()


def main3():  # Example usage of Problem 3
    sokoban_map = SokobanMap(10, 10, 3)
    sokoban_map.set_seed(13)  # tested solvable situation
    sokoban_map.mode = "eat"
    sokoban_map.mode2 = "1to1"

    sokoban_map.regenerate_map()
    gui = SokobanGUI(sokoban_map.get_map())
    gui.draw_map()
    searcher = SokobanSearcher(sokoban_map)
    while (state := searcher.astar()) is None:
        print("No solution found")
        sokoban_map.print_map()
        sokoban_map.regenerate_map()
    for direction in state.trajectory:
        sokoban_map.make_move(direction)
        gui.window_after(sokoban_map.get_map())
    gui.mainloop()


if __name__ == "__main__":
    main1()
    main2()
    main3()
