import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys

class AmnyamEnvModified(gym.Env):
    metadata = {"render_modes": ["human", "silent", "pygame"], "render_fps": 10}    
    
    MAX_AGE = 12  # Maximum age a fruit can reach
    PERFECT_AGE = 6  # The age at which eating a fruit gives maximum reward
    MAX_EPISODE_STEPS = 200  # Increased steps due to larger map
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.grid_height = 5
        self.grid_width = 100
        # 5 possible actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: 5 channels
        # Channel 0: Agent position (1 where agent is, 0 elsewhere)
        # Channel 1: Fruit age (normalized between 0 and 1)
        # Channel 2: Perfect age target (constant value)
        # Channel 3: Distance to each fruit (normalized)
        # Channel 4: Age difference from perfect age (normalized)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_height, self.grid_width, 5),
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
            self.cell_size = 10  # Smaller cells due to larger map
            self.window_size = (self.grid_width * self.cell_size, self.grid_height * self.cell_size)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Amnyam Modified Environment")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.RED = (255, 0, 0)
            self.GREEN = (0, 255, 0)
            self.BLUE = (0, 0, 255)

    def _get_observation(self):
        # Create multi-channel observation
        obs = np.zeros((self.grid_height, self.grid_width, 5), dtype=np.float32)
        
        # Channel 0: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1
        
        # Channel 1: Normalized fruit age
        # Channel 2: Perfect age target
        # Channel 3: Distance to each fruit
        # Channel 4: Age difference from perfect age
        for pos, fruit_info in self.fruits.items():
            # Channel 1: Fruit age
            obs[pos[0], pos[1], 1] = fruit_info['age'] / self.MAX_AGE
            
            # Channel 2: Perfect age target
            obs[pos[0], pos[1], 2] = self.PERFECT_AGE / self.MAX_AGE
            
            # Channel 3: Distance to agent (normalized by max possible distance)
            distance = np.sqrt((pos[0] - self.agent_pos[0])**2 + (pos[1] - self.agent_pos[1])**2)
            max_distance = np.sqrt(self.grid_height**2 + self.grid_width**2)
            obs[pos[0], pos[1], 3] = distance / max_distance
            
            # Channel 4: Age difference from perfect age (normalized)
            age_diff = abs(self.PERFECT_AGE - fruit_info['age'])
            obs[pos[0], pos[1], 4] = age_diff / self.PERFECT_AGE
        
        return obs

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)
        
        # Place agent at fixed starting position (middle-left)
        self.agent_pos = (self.grid_height // 2, 0)
        
        # Initialize fruits dictionary
        self.fruits = {}
        
        # Calculate time to reach each position from start
        # Assuming agent moves at 1 step per time unit
        for i in range(3):
            # First group of fruits (early in the path)
            pos = (self.grid_height // 2, 10 + i * 5)
            distance = pos[1]  # Distance from start
            # Set initial age so it reaches PERFECT_AGE when agent arrives
            initial_age = max(0, self.PERFECT_AGE - distance)
            self.fruits[pos] = {'age': initial_age}
        
        for i in range(3):
            # Second group (middle of the path)
            pos = (self.grid_height // 2, 40 + i * 5)
            distance = pos[1]
            initial_age = max(0, self.PERFECT_AGE - distance)
            self.fruits[pos] = {'age': initial_age}
        
        for i in range(3):
            # Third group (late in the path)
            pos = (self.grid_height // 2, 70 + i * 5)
            distance = pos[1]
            initial_age = max(0, self.PERFECT_AGE - distance)
            self.fruits[pos] = {'age': initial_age}
        
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
        
        # Convert action to direction
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # up, right, down, left, stay
        dx, dy = directions[action]
        
        # Calculate new position
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Check if new position is valid
        if 0 <= new_x < self.grid_height and 0 <= new_y < self.grid_width:
            # Update agent position
            self.agent_pos = (new_x, new_y)
            
            # Check if agent is on a fruit
            if self.agent_pos in self.fruits:
                fruit_info = self.fruits[self.agent_pos]
                # Calculate reward based on how close the age is to PERFECT_AGE
                reward += 1 - abs(self.PERFECT_AGE - fruit_info['age']) / self.PERFECT_AGE
                # Remove fruit after eating
                del self.fruits[self.agent_pos]
        
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
            for i in range(self.grid_height):
                for j in range(self.grid_width):
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
            for i in range(self.grid_height + 1):
                pygame.draw.line(self.screen, self.BLACK, 
                               (0, i * self.cell_size), 
                               (self.grid_width * self.cell_size, i * self.cell_size))
            for j in range(self.grid_width + 1):
                pygame.draw.line(self.screen, self.BLACK, 
                               (j * self.cell_size, 0), 
                               (j * self.cell_size, self.grid_height * self.cell_size))
            
            # Draw agent
            agent_rect = pygame.Rect(
                self.agent_pos[1] * self.cell_size + 1,
                self.agent_pos[0] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
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
                    pos[1] * self.cell_size + 2,
                    pos[0] * self.cell_size + 2,
                    self.cell_size - 4,
                    self.cell_size - 4
                )
                pygame.draw.rect(self.screen, color, fruit_rect)
                
                # Draw age number
                font = pygame.font.Font(None, 12)
                text = font.render(str(int(age)), True, self.BLACK)
                text_rect = text.get_rect(center=(
                    pos[1] * self.cell_size + self.cell_size // 2,
                    pos[0] * self.cell_size + self.cell_size // 2
                ))
                self.screen.blit(text, text_rect)
            
            # Draw step counter
            font = pygame.font.Font(None, 24)
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