import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
    from Tkinter import PhotoImage
else:
    import tkinter as tk
    from tkinter import PhotoImage


UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()

        # space definitions
        self.action_space = [0, 1, 2, 3]  # ['u', 'd', 'l', 'r'] # 决策空间
        self.n_actions = len(self.action_space)
        self.state_space = [(i * UNIT + UNIT/2, j * UNIT + UNIT/2) for i in range(MAZE_H) for j in range(MAZE_W)]

        # Creating Canvas
        self.title('Q-learning')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT/2, UNIT/2])
        
        self.bm_stone = PhotoImage(file="obstacles.png")
        self.stone1 = self.canvas.create_image(origin[0]+UNIT * 4, origin[1]+UNIT,image=self.bm_stone)
        self.stone2 = self.canvas.create_image(origin[0]+UNIT, origin[1]+UNIT * 4,image=self.bm_stone)
        self.stone3 = self.canvas.create_image(origin[0]+UNIT*4, origin[1]+UNIT * 3,image=self.bm_stone)
        self.stone4 = self.canvas.create_image(origin[0]+UNIT*3, origin[1]+UNIT * 4,image=self.bm_stone)

        self.bm_yoki = PhotoImage(file="character.png")
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)

        self.bm_Candy = PhotoImage(file="candy.png")
        self.Candy = self.canvas.create_image(origin[0]+4*UNIT, origin[1]+4*UNIT,image=self.bm_Candy)

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.yoki)
        origin = np.array([UNIT/2, UNIT/2])
        
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)
        return tuple(self.canvas.coords(self.yoki))

    def step(self, action):
        s = self.canvas.coords(self.yoki)
        base_action = np.array([0, 0])
        if action == 0:   # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.yoki, base_action[0], base_action[1]) 
        s_ = self.canvas.coords(self.yoki)

        # 回报函数
        if s_ == self.canvas.coords(self.Candy):
            # reward = 1
            reward = 1000
            done = True
            # s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.stone1), self.canvas.coords(self.stone2),self.canvas.coords(self.stone3),self.canvas.coords(self.stone4)]:
            # reward = -1
            reward = -1000
            done = True
            # s_ = 'terminal'
        else:
            # reward = 0
            reward = -((s_[0] - self.canvas.coords(self.Candy)[0]) ** 2 + (s_[1] - self.canvas.coords(self.Candy)[1]) ** 2) // 1e4  # in -50~0
            done = False

        return tuple(s_), reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

    def update_title(self, string: str):
        self.title(string)

