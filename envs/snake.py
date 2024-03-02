import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, clear_output

class SnakeEnv:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(np.random.randint(2, self.grid_size[0] - 2), np.random.randint(2, self.grid_size[1] - 2))]
        self.food = None
        self.direction = (0, 1)  # Moving right initially
        self.score = 0
        self._place_food()
        return self._get_observation()

    def _place_food(self):
        while self.food is None or self.food in self.snake:
            self.food = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))

    def step(self, action):
        # Define action as a direction change: 0=left, 1=right, 2=up, 3=down
        direction_changes = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.direction = direction_changes[action]
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for game over conditions
        if new_head in self.snake or new_head[0] < 0 or new_head[1] < 0 or new_head[0] >= self.grid_size[0] or new_head[1] >= self.grid_size[1]:
            return self._get_observation(), 0, True, {}  # Game over

        self.snake.insert(0, new_head)

        # Check for food consumption
        if new_head == self.food:
            self.score += 1
            self._place_food()
            reward = 1
        else:
            self.snake.pop()
            reward = 0

        return self._get_observation(), reward, False, {}
    
    def hat(self, vec):
        return np.array(np.array(vec) / math.sqrt(vec[0] ** 2 + vec[1] ** 2))

    def _get_observation(self):

        pos = self.hat([self.food[0] - self.snake[0][0], self.food[1] - self.snake[0][1]])
        dirc = np.array(self.direction)

        obs = [0, 0, 0, 0]
        
        # up
        if np.all(pos - dirc == 0):
            obs[0] = 1
        
        # Down
        if np.all(pos + dirc == 0):
            obs[1] = 1
            
        if np.all(pos * dirc == 0):
            # Right
            if pos[0] * dirc[1] > 0:
                obs[2] = 1
            else:
                obs[3] = 1
        
        return obs
            
        
        
#         # get the position of the unit y vector snake ref frame
#         j_hat = (self.direction[0], -1 * self.direction[1])
#         i_hat = (-1 * self.direction[1], -1 * self.direction[0])
# # #         print(self.direction, self.snake[0], i_hat, j_hat)
# #         print("snake head: ", self.snake[0])
# #         print("snake dir: ", self.direction)
#         obs = []
#         for i in range(-5, 6):
#             if i != 0:
#                 pos = ( self.snake[0][0] + i * i_hat[0], self.snake[0][1] + i * i_hat[1])
#                 if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
#                     if pos == self.food:
#                         obs.append(1)
#                     elif pos in self.snake:
#                         obs.append(-1)
#                     else:
#                         obs.append(0)
#                 else:
#                     obs.append(-1)
#         for i in range(-1, 2):
#             pos = ( self.snake[0][0] + j_hat[0] + i * i_hat[0], self.snake[0][1] + j_hat[1] + i * i_hat[1])
#             if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
#                 if pos == self.food:
#                     obs.append(1)
#                 elif pos in self.snake:
#                     obs.append(-1)
#                 else:
#                     obs.append(0)
#             else:
#                 obs.append(-1)
#         return np.array(obs)

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
#         ax.set_xticks(range(self.grid_size[0]))
#         ax.set_yticks(range(self.grid_size[1]))
        ax.grid(False)
        plt.axis('off')
        # Plot snake
        for part in self.snake:
            ax.add_patch(plt.Rectangle(part, 1, 1, color="green"))
        # Plot food
        ax.add_patch(plt.Rectangle(self.food, 1, 1, color="red"))
#         display(fig)
#         clear_output(wait=True)
#         plt.close()