from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from maze_env import Maze, MAZE_H, MAZE_W

MAX_STEPS = MAZE_H * MAZE_W * 10


if __name__ == '__main__':
    env = Maze()

    action_trajectories = []

    # Q-learning parameters
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    epsilon_decay = 0.97
    episodes = 100
    record_rounds = 10

    env.canvas.coords(env.yoki)

    # Initialize Q-table
    q_table = {}
    for state in env.state_space:
        q_table[state] = {action: 0 for action in env.action_space}  # action 0, 1, 2, 3

    # Action choosing scheme based on Q-table, called before an action
    def choose_action(state: Tuple[float, float]):
        if np.random.rand() < epsilon:
            return np.random.choice([action for action in env.action_space])
        else:
            state_actions = q_table[state]
            return max(state_actions, key=state_actions.get)

    # Q-table updating scheme based on reward trajectory, called after an action
    def update_q_table(state: Tuple[float, float], action: int, reward: int, next_state: Tuple[float, float]):
        best_next_action = max(q_table[next_state], key=q_table[next_state].get)
        td_target = reward + discount_factor * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += learning_rate * td_error


    # Training the agent
    for episode in tqdm(range(episodes + 1)):
        state = env.reset()
        epsilon *= epsilon_decay
        action_tr = []

        for _ in range(MAX_STEPS):  # Truncate at MAX_STEPS
            action = choose_action(state)
            action_tr.append(action)
            next_state, reward, done = env.step(action)
            update_q_table(state, action, reward, next_state)
            state = next_state
            if done:
                break

        if episode % (episodes // record_rounds) == 0:
            action_trajectories.append(action_tr)
            print()
            print(pd.DataFrame(q_table).T)


    def update(action_trs: list):
        # 更新图形化界面
        for index, tr in enumerate(action_trs):
            s = env.reset()
            env.update_title(f"Episode {index * (episodes // record_rounds)}")
            for a in tr:
                env.render()
                s, r, done = env.step(a)
                if done:
                    break

    env.after(100, update, action_trajectories)
    env.mainloop()
