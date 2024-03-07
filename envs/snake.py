import matplotlib.pyplot as plt
import numpy as np
import math

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
        reward = 0
        if new_head == self.food:
            self.score += 1
            self._place_food()
            reward = 1
        else:
            self.snake.pop()

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

#     def _get_observation(self):

#         pos = self.hat([self.food[0] - self.snake[0][0], self.food[1] - self.snake[0][1]])
#         dirc = np.array(self.direction)

#         obs = list(pos) + list(dirc)
        
#         return obs

    def render_save(self, plot_path: str, ax=None):
        
        plt.cla()
        if ax == None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        plt.axis('off')
        
        # Plot snake
        for part in self.snake:
            ax.add_patch(plt.Rectangle(part, 1, 1, color="green"))
            
        # Plot food
        ax.add_patch(plt.Rectangle(self.food, 1, 1, color="red"))  
        plt.tight_layout()
        plt.savefig(plot_path)
