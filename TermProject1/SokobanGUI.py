
import tkinter as tk
from PIL import Image, ImageTk
from typing import List
from copy import deepcopy
from Utilities import State
from os import path

# Constants
PIC_HEIGHT = 32
PIC_WIDTH = 32
IMG_PATH = "./assets/"
IMG_FILENAME = {
    State.FREE: "free.png",
    State.WALL: "wall.png",
    State.BOX: "box.png",
    State.GOAL: "goal.png",
    State.PLAYER: "player.png",
    State.BOX_ON_GOAL: "box_on_goal.png",
    State.PLAYER_ON_GOAL: "player.png"
}


class SokobanGUI:
    def __init__(self, sokoban_map: List[List[State]]):
        self.sokoban_map = sokoban_map
        # tkinter setup
        self.window = tk.Tk()
        self.window.title("Sokoban")
        self.window.geometry(f"{len(sokoban_map[0]) * PIC_WIDTH}x{len(sokoban_map) * PIC_HEIGHT}")
        self.canvas = tk.Canvas(self.window,
                                width=len(sokoban_map[0]) * PIC_WIDTH, height=len(sokoban_map) * PIC_HEIGHT)
        self.canvas.pack()
        # loading images
        self.images = {}
        self.load_images()
        # for window.after
        self.clock = 0

    def load_images(self):
        for state in State:
            image_path = path.join(IMG_PATH, IMG_FILENAME[state])
            image = Image.open(image_path)
            image = image.resize((PIC_WIDTH, PIC_HEIGHT))
            photo = ImageTk.PhotoImage(image)
            self.images[state] = photo

    def draw_map(self):
        for y, row in enumerate(self.sokoban_map):
            for x, state in enumerate(row):
                self.canvas.create_image(x * PIC_WIDTH, y * PIC_HEIGHT, image=self.images[state], anchor=tk.NW)

    def update_map(self, sokoban_map: List[List[State]]):
        self.sokoban_map = sokoban_map
        self.canvas.delete("all")
        self.draw_map()

    def window_after(self, sokoban_map: List[List[State]], time: int = 500):
        sokoban_map = deepcopy(sokoban_map)
        self.clock += time
        self.window.after(self.clock, self.update_map, sokoban_map)

    def mainloop(self):
        self.window.mainloop()

