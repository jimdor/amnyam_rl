import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys

class AmnyamEnv(gym.Env):
    metadata = {"render_modes": ["human", "silent", "pygame"], "render_fps": 10}    
    """
    A custom environment that follows gymnasium interface.
    This is a simple environment where the agent needs to reach a target.
    """
    
    MAX_AGE = 12  # Maximum age a fruit can reach before disappearing
    PERFECT_AGE = 6  # The age at which eating a fruit gives maximum reward
    MAX_EPISODE_STEPS = 100
    
    def __init__(self, render_mode=None, grid_size=10):
        super().__init__()
        
        self.grid_size = grid_size
        # 6 possible actions: 0=stay, 1=up, 2=right, 3=down, 4=left, 5=eat
        self.action_space = spaces.Discrete(6)
        
        # Observation space: 5 channels
        # Channel 0: Agent position (1 where agent is, 0 elsewhere)
        # Channel 1: Fruit age (normalized between 0 and 1)
        # Channel 2: Perfect age target (constant value)
        # Channel 3: Distance to each fruit (normalized)
        # Channel 4: Age difference from perfect age (normalized)
        # Channel 5: Fruits within eating range (binary)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 6),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.agent_pos = None
        self.fruits = {}  # Dictionary to store fruit information: (x,y) -> {'age': float}
        self.steps = 0
        self.render_mode = render_mode
        
        # Pygame initialization
        if self.render_mode == "pygame":
            pygame.init()
            self.cell_size = 60
            self.window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Amnyam Environment")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.RED = (255, 0, 0)
            self.GREEN = (0, 255, 0)
            self.BLUE = (0, 0, 255)

    def _get_observation(self):
        # Create multi-channel observation
        obs = np.zeros((self.grid_size, self.grid_size, 6), dtype=np.float32)
        
        # Channel 0: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1
        
        # Channel 1: Normalized fruit age
        # Channel 2: Perfect age target
        # Channel 3: Distance to each fruit
        # Channel 4: Age difference from perfect age
        # Channel 5: Fruits within eating range
        for pos, fruit_info in self.fruits.items():
            # Channel 1: Fruit age
            obs[pos[0], pos[1], 1] = fruit_info['age'] / self.MAX_AGE
            
            # Channel 2: Perfect age target
            obs[pos[0], pos[1], 2] = self.PERFECT_AGE / self.MAX_AGE
            
            # Channel 3: Distance to agent (normalized by max possible distance)
            distance = np.sqrt((pos[0] - self.agent_pos[0])**2 + (pos[1] - self.agent_pos[1])**2)
            max_distance = np.sqrt(2 * (self.grid_size - 1)**2)  # Maximum possible distance
            obs[pos[0], pos[1], 3] = distance / max_distance
            
            # Channel 4: Age difference from perfect age (normalized)
            age_diff = abs(self.PERFECT_AGE - fruit_info['age'])
            obs[pos[0], pos[1], 4] = age_diff / self.PERFECT_AGE
            
            # Channel 5: Fruits within eating range
            if (abs(self.agent_pos[0] - pos[0]) <= 1 and 
                abs(self.agent_pos[1] - pos[1]) <= 1):
                obs[pos[0], pos[1], 5] = 1
        
        return obs

    def _spawn_fruit(self):
        # Randomly decide if we should spawn a new fruit (20% chance)
        if self.np_random.random() < 0.2:
            # Try to find an empty cell
            for _ in range(10):  # Try up to 10 times
                pos = self.np_random.integers(0, self.grid_size, size=2)
                # Check if position is not occupied by agent or other fruits
                if (pos[0] != self.agent_pos[0] or pos[1] != self.agent_pos[1]) and tuple(pos) not in self.fruits:
                    # Random initial age between 0 and 2
                    initial_age = self.np_random.uniform(0, 2)
                    self.fruits[tuple(pos)] = {'age': initial_age}
                    break

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)
        
        # Place agent at random position
        self.agent_pos = self.np_random.integers(0, self.grid_size, size=2)
        
        # Initialize fruits dictionary
        self.fruits = {}
        
        # Place initial fruits
        num_fruits = self.np_random.integers(3, 6)
        for _ in range(num_fruits):
            while True:
                pos = self.np_random.integers(0, self.grid_size, size=2)
                pos_tuple = tuple(pos)
                # Check if position is not occupied by agent or other fruits
                if (pos[0] != self.agent_pos[0] or pos[1] != self.agent_pos[1]) and pos_tuple not in self.fruits:
                    initial_age = self.np_random.uniform(0, 2)
                    self.fruits[pos_tuple] = {'age': initial_age}
                    break
        
        self.steps = 0
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        if action == 0:  # Stay
            pass
        elif action == 5:  # Eat
            # Check if there's a fruit within 1 cell radius
            for fruit_pos in list(self.fruits.keys()):
                if (abs(self.agent_pos[0] - fruit_pos[0]) <= 1 and 
                    abs(self.agent_pos[1] - fruit_pos[1]) <= 1):
                    fruit_info = self.fruits[fruit_pos]
                    # Calculate reward based on how close the age is to PERFECT_AGE
                    reward += 1 - abs(self.PERFECT_AGE - fruit_info['age']) / self.PERFECT_AGE
                    # reward += 1 - abs(self.PERFECT_AGE - fruit_info['age'])
                    # Remove fruit after eating
                    del self.fruits[fruit_pos]
        else:  # Movement actions (1-4)
            # Convert action to direction
            directions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]  # stay, up, right, down, left
            dx, dy = directions[action]
            
            # Calculate new position
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy
            
            # Check if new position is valid
            if (0 <= new_x < self.grid_size and 
                0 <= new_y < self.grid_size and 
                (new_x, new_y) not in self.fruits):  # Can't move onto fruit
                
                # Update agent position
                self.agent_pos = (new_x, new_y)
        
        # Update fruit ages
        fruits_to_remove = []
        for pos, fruit_info in self.fruits.items():
            fruit_info['age'] += 1
            if fruit_info['age'] >= self.MAX_AGE:
                fruits_to_remove.append(pos)
        
        # Remove old fruits
        for pos in fruits_to_remove:
            del self.fruits[pos]
        
        # Try to spawn new fruits
        self._spawn_fruit()
        
        # Update step counter
        self.steps += 1
        
        # Check if episode should end
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment to the screen
        """
        if self.render_mode == "human":
            # Get current observation
            obs = self._get_observation()
            
            # Simple ASCII rendering with fruit ages
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if obs[i, j, 0] == 1:  # Agent
                        print("A", end=" ")
                    elif obs[i, j, 1] > 0:  # Fruit
                        age = int(obs[i, j, 1] * self.MAX_AGE)
                        print(str(age), end=" ")
                    else:
                        print(".", end=" ")
                print()
            print(f"Steps: {self.steps}/{self.MAX_EPISODE_STEPS}")
            print()
        
        elif self.render_mode == "pygame":
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw grid
            for i in range(self.grid_size + 1):
                pygame.draw.line(self.screen, self.BLACK, 
                               (i * self.cell_size, 0), 
                               (i * self.cell_size, self.window_size))
                pygame.draw.line(self.screen, self.BLACK, 
                               (0, i * self.cell_size), 
                               (self.window_size, i * self.cell_size))
            
            # Draw agent
            agent_rect = pygame.Rect(
                self.agent_pos[1] * self.cell_size + 5,
                self.agent_pos[0] * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )
            pygame.draw.rect(self.screen, self.BLUE, agent_rect)
            
            # Draw fruits
            for pos, fruit_info in self.fruits.items():
                age = fruit_info['age']
                # Color gradient from green (young) to red (old)
                color = (
                    int(255 * (age / self.MAX_AGE)),
                    int(255 * (1 - age / self.MAX_AGE)),
                    0
                )
                fruit_rect = pygame.Rect(
                    pos[1] * self.cell_size + 10,
                    pos[0] * self.cell_size + 10,
                    self.cell_size - 20,
                    self.cell_size - 20
                )
                pygame.draw.rect(self.screen, color, fruit_rect)
                
                # Draw age number
                font = pygame.font.Font(None, 24)
                text = font.render(str(int(age)), True, self.BLACK)
                text_rect = text.get_rect(center=(
                    pos[1] * self.cell_size + self.cell_size // 2,
                    pos[0] * self.cell_size + self.cell_size // 2
                ))
                self.screen.blit(text, text_rect)
            
            # Draw step counter
            font = pygame.font.Font(None, 36)
            text = font.render(f"Steps: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.BLACK)
            self.screen.blit(text, (10, 10))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """
        Clean up resources
        """
        if self.render_mode == "pygame":
            pygame.quit() 