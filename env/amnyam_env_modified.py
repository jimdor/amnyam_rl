import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys
import os

class AmnyamEnvModified(gym.Env):
    metadata = {"render_modes": ["human", "silent", "pygame"], "render_fps": 10}    
    
    MAX_AGE = 250  # Maximum age a fruit can reach
    PERFECT_AGE = 150  # The age at which eating a fruit gives maximum reward
    MAX_EPISODE_STEPS = 300  # Increased steps due to larger map
    
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
        self.np_random = None  # Will be initialized in reset
        
        # Pygame initialization
        if self.render_mode == "pygame":
            pygame.init()
            self.cell_size = 90  # Increased from 30 to 90 (3x bigger)
            
            # Camera view settings
            self.view_width = 15  # Number of cells visible horizontally
            self.view_height = 5  # Number of cells visible vertically
            self.window_size = (self.view_width * self.cell_size + 60,  # Increased padding
                              self.view_height * self.cell_size + 180)  # Increased padding
            
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Amnyam Modified Environment")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.RED = (255, 0, 0)
            self.GREEN = (0, 255, 0)
            self.BLUE = (0, 0, 255)
            self.GRAY = (200, 200, 200)  # For grid lines
            
            # Load images
            try:
                image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'amnyam.png')
                self.agent_image = pygame.image.load(image_path)
                self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size - 12, self.cell_size - 12))  # Adjusted image size
            except pygame.error as e:
                print(f"Error loading image: {e}")
                self.agent_image = None

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
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Place agent at fixed starting position (middle-left)
        self.agent_pos = (self.grid_height // 2, 0)
        
        # Initialize fruits dictionary
        self.fruits = {}
        
        # Calculate time to reach each position from start
        # Assuming agent moves at 1 step per time unit
        for i in range(3):
            # First group of fruits (early in the path)
            base_x = self.grid_height // 2
            base_y = 10 + i * 5
            # Add small random variation to position
            x = max(0, min(self.grid_height - 1, base_x + self.np_random.randint(-1, 2)))
            y = max(0, min(self.grid_width - 1, base_y + self.np_random.randint(-2, 3)))
            pos = (x, y)
            distance = y  # Distance from start
            # Add small random variation to initial age
            age_variation = self.np_random.uniform(-5, 5)
            initial_age = max(0, self.PERFECT_AGE - distance + age_variation)
            self.fruits[pos] = {'age': initial_age}
        
        for i in range(3):
            # Second group (middle of the path)
            base_x = self.grid_height // 2
            base_y = 40 + i * 5
            # Add small random variation to position
            x = max(0, min(self.grid_height - 1, base_x + self.np_random.randint(-1, 2)))
            y = max(0, min(self.grid_width - 1, base_y + self.np_random.randint(-2, 3)))
            pos = (x, y)
            distance = y
            # Add small random variation to initial age
            age_variation = self.np_random.uniform(-5, 5)
            initial_age = max(0, self.PERFECT_AGE - distance - 10 + age_variation)
            self.fruits[pos] = {'age': initial_age}
        
        for i in range(3):
            # Third group (late in the path)
            base_x = self.grid_height // 2
            base_y = 70 + i * 5
            # Add small random variation to position
            x = max(0, min(self.grid_height - 1, base_x + self.np_random.randint(-1, 2)))
            y = max(0, min(self.grid_width - 1, base_y + self.np_random.randint(-2, 3)))
            pos = (x, y)
            distance = y
            # Add small random variation to initial age
            age_variation = self.np_random.uniform(-5, 5)
            initial_age = max(0, self.PERFECT_AGE - distance - 20 + age_variation)
            self.fruits[pos] = {'age': initial_age}
            
        # Add fruit in the right with some randomness
        base_x = self.grid_height // 2
        base_y = self.grid_width - 1
        # Add small random variation to position
        x = max(0, min(self.grid_height - 1, base_x + self.np_random.randint(-1, 2)))
        y = max(0, min(self.grid_width - 1, base_y + self.np_random.randint(-2, 3)))
        corner_pos = (x, y)
        # Add small random variation to initial age
        age_variation = self.np_random.uniform(-2, 2)
        initial_age = max(0, 1 + age_variation)
        self.fruits[corner_pos] = {'age': initial_age}
        
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

        # Age existing fruits
        for key in self.fruits.keys():
            self.fruits[key]['age'] += 1
        
        # Check if episode should end
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            
        # Terminate if all fruits are eaten
        if len(self.fruits) == 0:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment to the screen.
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
            
            # Calculate camera position (centered on agent)
            camera_x = max(0, min(self.grid_width - self.view_width, 
                                self.agent_pos[1] - self.view_width // 2))
            camera_y = max(0, min(self.grid_height - self.view_height, 
                                self.agent_pos[0] - self.view_height // 2))
            
            # Draw grid with thicker lines
            for i in range(self.view_height + 1):
                pygame.draw.line(self.screen, self.GRAY, 
                               (0, i * self.cell_size), 
                               (self.view_width * self.cell_size, i * self.cell_size),
                               6)  # Increased line thickness
            for j in range(self.view_width + 1):
                pygame.draw.line(self.screen, self.GRAY, 
                               (j * self.cell_size, 0), 
                               (j * self.cell_size, self.view_height * self.cell_size),
                               6)  # Increased line thickness
            
            # Draw fruits in view
            for pos, fruit_info in self.fruits.items():
                # Check if fruit is in view
                if (camera_x <= pos[1] < camera_x + self.view_width and 
                    camera_y <= pos[0] < camera_y + self.view_height):
                    # Calculate screen position
                    screen_x = pos[1] - camera_x
                    screen_y = pos[0] - camera_y
                    
                    age = fruit_info['age']
                    # Color transition based on age relative to perfect age
                    if age < self.PERFECT_AGE:
                        # Transition from yellow to green
                        progress = age / self.PERFECT_AGE
                        color = (
                            int(255 * (1 - progress)),  # R decreases from 255 to 0
                            int(255),                   # G stays at max
                            0                           # B stays 0
                        )
                    else:
                        # Transition from green to red
                        progress = (age - self.PERFECT_AGE) / (self.MAX_AGE - self.PERFECT_AGE)
                        color = (
                            int(255 * progress),        # R increases from 0 to 255
                            int(255 * (1 - progress)),  # G decreases from 255 to 0
                            0                           # B stays 0
                        )
                    fruit_rect = pygame.Rect(
                        screen_x * self.cell_size + 9,  # Increased padding
                        screen_y * self.cell_size + 9,  # Increased padding
                        self.cell_size - 18,  # Increased padding
                        self.cell_size - 18   # Increased padding
                    )
                    pygame.draw.rect(self.screen, color, fruit_rect)
                    
                    # Draw age number with larger font
                    font = pygame.font.Font(None, 60)  # Increased font size
                    text = font.render(str(int(age)), True, self.BLACK)
                    text_rect = text.get_rect(center=(
                        screen_x * self.cell_size + self.cell_size // 2,
                        screen_y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
            
            # Draw agent using image if available, otherwise use rectangle
            if self.agent_image:
                screen_x = self.agent_pos[1] - camera_x
                screen_y = self.agent_pos[0] - camera_y
                self.screen.blit(self.agent_image, 
                               (screen_x * self.cell_size + 6,  # Increased padding
                                screen_y * self.cell_size + 6))  # Increased padding
            else:
                screen_x = self.agent_pos[1] - camera_x
                screen_y = self.agent_pos[0] - camera_y
                agent_rect = pygame.Rect(
                    screen_x * self.cell_size + 6,  # Increased padding
                    screen_y * self.cell_size + 6,  # Increased padding
                    self.cell_size - 12,  # Increased padding
                    self.cell_size - 12   # Increased padding
                )
                pygame.draw.rect(self.screen, self.BLUE, agent_rect)
            
            # Draw step counter with larger font and better positioning
            font = pygame.font.Font(None, 108)  # Increased font size
            text = font.render(f"Steps: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.BLACK)
            self.screen.blit(text, (30, self.view_height * self.cell_size + 30))  # Increased padding
            
            # Draw position indicator
            pos_text = font.render(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})", True, self.BLACK)
            self.screen.blit(pos_text, (30, self.view_height * self.cell_size + 120))  # Increased padding
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """
        Clean up resources
        """
        if self.render_mode == "pygame":
            pygame.quit() 