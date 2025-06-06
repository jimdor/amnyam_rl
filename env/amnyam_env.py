import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys
import os
from collections import deque


class AmnyamEnv(gym.Env):
    metadata = {"render_modes": ["human", "silent", "pygame"], "render_fps": 5}

    # Available observation channels
    CHANNEL_NAMES = {
        0: "agent_position",
        1: "fruit_locations",
        2: "fruit_age",
        3: "perfect_age_target",
        4: "distance_to_fruits",
        5: "age_difference_from_perfect",
        6: "fruit_expiration_urgency",
        7: "optimal_harvest_timing",
        8: "agent_position_history",
        9: "episode_progress",
    }

    def __init__(self,
                 render_mode=None,
                 observation_channels=(0, 1, 2, 3, 4),  # Which channels to include
                 max_episode_steps=100,
                 grid_size=(10, 10),  # (height, width) or single int for square
                 fruit_spawning=('random', 1.0),  # ('mode', temperature/group_count)
                 seed=None):
        super().__init__()

        # Store configuration
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.observation_channels = observation_channels
        self.fruit_spawning_mode, self.fruit_spawning_param = fruit_spawning

        # Validate observation channels
        for ch in observation_channels:
            if ch not in self.CHANNEL_NAMES:
                raise ValueError(f"Unknown observation channel: {ch}. Available: {list(self.CHANNEL_NAMES.keys())}")

        # Parse grid size
        if isinstance(grid_size, int):
            self.grid_height = self.grid_width = grid_size
        else:
            self.grid_height, self.grid_width = grid_size

        # Configure fruit aging based on spawning mode and parameter
        if self.fruit_spawning_mode == 'random':
            # Temperature affects spawn rate and fruit aging
            temperature = max(0.1, self.fruit_spawning_param)  # Ensure positive
            self.MAX_AGE = max(5, int(12 * temperature))
            self.PERFECT_AGE = max(2, int(6 * temperature))
            self.spawn_probability = min(0.5, 0.1 * temperature)
        elif self.fruit_spawning_mode == 'strategic':
            # Group count affects strategic placement and aging
            group_count = max(1, int(self.fruit_spawning_param))
            self.MAX_AGE = max(50, self.grid_width * 2)
            self.PERFECT_AGE = max(25, self.grid_width + 10)
            self.fruit_groups = group_count
        else:
            raise ValueError(f"Unknown fruit spawning mode: {self.fruit_spawning_mode}. Use 'random' or 'strategic'.")

        # 5 possible actions: 0=stay, 1=up, 2=right, 3=down, 4=left
        self.action_space = spaces.Discrete(5)

        # Observation space: configurable channels
        num_channels = len(observation_channels)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_height, self.grid_width, num_channels),
            dtype=np.float32
        )

        # Initialize state
        self.agent_pos = None
        self.fruits = {}  # Dictionary: (x,y) -> {'age': float}
        self.steps = 0
        self.render_mode = render_mode
        self.agent_position_history = deque(maxlen=10)

        if self.render_mode == 'human':
            self.render_digits_len = len(str(self.MAX_AGE))
        else:
            self.render_digits_len = None

        # Initialize pygame if needed
        if self.render_mode == "pygame":
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        self.cell_size = 80

        # Camera view settings for large grids
        if self.grid_width > 15 or self.grid_height > 15:
            self.use_camera = True
            self.view_width = min(15, self.grid_width)
            self.view_height = min(10, self.grid_height)
        else:
            self.use_camera = False
            self.view_width = self.grid_width
            self.view_height = self.grid_height

        self.window_size = (
            self.view_width * self.cell_size + 60,
            self.view_height * self.cell_size + 200
        )

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Amnyam Smart-Agile Environment")
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)

        # Load amnyam image
        try:
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'amnyam.png')
            self.agent_image = pygame.image.load(image_path)
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size - 12, self.cell_size - 12))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Could not load agent image: {e}")
            self.agent_image = None

    def _get_observation(self):
        """Generate observation with only requested channels"""
        num_channels = len(self.observation_channels)
        obs = np.zeros((self.grid_height, self.grid_width, num_channels), dtype=np.float32)

        # Calculate all possible channels, then extract requested ones
        full_obs = np.zeros((self.grid_height, self.grid_width, len(self.CHANNEL_NAMES)), dtype=np.float32)

        # Channel 0: Agent position
        full_obs[self.agent_pos[0], self.agent_pos[1], 0] = 1.0

        # Process fruits for other channels
        for pos, fruit_info in self.fruits.items():
            age = fruit_info['age']

            # Channel 1: Fruit locations
            full_obs[pos[0], pos[1], 1] = 1.0

            # Channel 2: Normalized fruit age
            full_obs[pos[0], pos[1], 2] = min(1.0, age / self.MAX_AGE)

            # Channel 3: Perfect age target (constant)
            full_obs[pos[0], pos[1], 3] = self.PERFECT_AGE / self.MAX_AGE

            # Channel 4: Manhattan distance to agent (normalized)
            distance = abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1])
            max_distance = self.grid_height + self.grid_width - 2
            full_obs[pos[0], pos[1], 4] = distance / max_distance

            # Channel 5: Age difference from perfect age (normalized)
            age_diff = abs(self.PERFECT_AGE - age)
            full_obs[pos[0], pos[1], 5] = min(1.0, age_diff / self.PERFECT_AGE)

            # Channel 6: Fruit expiration urgency
            steps_to_expire = max(0, self.MAX_AGE - age)
            full_obs[pos[0], pos[1], 6] = steps_to_expire / self.MAX_AGE

            # Channel 7: Optimal harvest timing
            steps_to_perfect = self.PERFECT_AGE - age
            if steps_to_perfect > 0:
                full_obs[pos[0], pos[1], 7] = steps_to_perfect / self.PERFECT_AGE
            else:
                full_obs[pos[0], pos[1], 7] = 0.0

        # Channel 8: Agent position history
        for steps_ago, pos in enumerate(reversed(self.agent_position_history)):
            if steps_ago > 0:  # Skip current position (steps_ago = 0)
                history_value = 1.0 - 0.1 * steps_ago
                if history_value > 0:  # Only add positive values
                    # Add to existing value (in case agent visited same position multiple times)
                    full_obs[pos[0], pos[1], 8] = max(full_obs[pos[0], pos[1], 8], history_value)

        # Channel 9: Episode progress (global temporal information)
        episode_progress = self.steps / self.max_episode_steps
        full_obs[:, :, 9] = episode_progress

        # Extract only requested channels
        for i, channel_idx in enumerate(self.observation_channels):
            obs[:, :, i] = full_obs[:, :, channel_idx]

        return obs

    def _spawn_fruit_random(self):
        """Random fruit spawning controlled by temperature"""
        if self.np_random.random() < self.spawn_probability:
            # Try to find an empty cell
            for _ in range(100):
                pos = (self.np_random.integers(0, self.grid_height),
                       self.np_random.integers(0, self.grid_width))
                if pos != tuple(self.agent_pos) and pos not in self.fruits:
                    initial_age = self.np_random.uniform(0, min(3, self.PERFECT_AGE / 2))
                    self.fruits[pos] = {'age': initial_age}
                    break

    def _spawn_fruit_strategic(self):
        """Strategic fruit placement in groups"""
        self.fruits = {}

        # Distribute groups along the width
        for group in range(self.fruit_groups):
            # Calculate group center position
            group_y = (group + 1) * self.grid_width // (self.fruit_groups)
            group_x = self.grid_height // 2

            # Add fruits around this center
            fruits_per_group = 2
            max_attempts_per_fruit = 10  # Prevent infinite loops

            for i in range(fruits_per_group):
                placed = False
                attempts = 0

                while not placed and attempts < max_attempts_per_fruit:
                    # Add positional variation
                    x = max(0, min(self.grid_height - 1,
                                   group_x + self.np_random.integers(-2, 2)))
                    y = max(0, min(self.grid_width - 1,
                                   group_y + self.np_random.integers(-2, 2)))
                    pos = (x, y)

                    if pos != tuple(self.agent_pos) and pos not in self.fruits:
                        # Strategic aging: fruits should be ready when agent arrives
                        time_to_reach = abs(group_y - self.agent_pos[1])  # Assuming start at x=0
                        age_variation = self.np_random.integers(-3, 3)
                        initial_age = max(0, self.PERFECT_AGE - time_to_reach + age_variation)
                        self.fruits[pos] = {'age': initial_age}
                        placed = True

                    attempts += 1

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is None:
            seed = self.seed
        super().reset(seed=seed)

        # Agent placement strategy
        if self.fruit_spawning_mode == 'strategic':
            # Start at left edge for horizontal strategic layout
            self.agent_pos = np.array([self.grid_height // 2, 0])
        else:
            # Random placement
            self.agent_pos = np.array([
                self.np_random.integers(0, self.grid_height),
                self.np_random.integers(0, self.grid_width)
            ])

        self.agent_position_history.clear()
        self.agent_position_history.append(tuple(self.agent_pos))

        # Initialize fruits based on mode
        if self.fruit_spawning_mode == 'random':
            self.fruits = {}
            # Place initial random fruits
            num_initial_fruits = self.np_random.integers(1, min(8, (self.grid_height * self.grid_width) // 10))
            for _ in range(num_initial_fruits):
                for _ in range(100):  # Try up to 100 times
                    pos = (self.np_random.integers(0, self.grid_height),
                           self.np_random.integers(0, self.grid_width))
                    if pos != tuple(self.agent_pos) and pos not in self.fruits:
                        initial_age = self.np_random.uniform(0, min(3, self.PERFECT_AGE / 2))
                        self.fruits[pos] = {'age': initial_age}
                        break
        elif self.fruit_spawning_mode == 'strategic':
            self._spawn_fruit_strategic()

        self.steps = 0
        return self._get_observation(), {}

    def step(self, action):
        """Execute one environment step"""
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Handle movement
        if action == 0:  # Stay
            pass
        else:  # Movement: 1=up, 2=right, 3=down, 4=left
            directions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
            dx, dy = directions[action]

            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy

            # Check bounds
            if 0 <= new_x < self.grid_height and 0 <= new_y < self.grid_width:
                new_pos = (new_x, new_y)

                # Check for fruit consumption
                if new_pos in self.fruits:
                    fruit_age = self.fruits[new_pos]['age']
                    # Reward based on proximity to perfect age
                    age_score = 1.0 - abs(self.PERFECT_AGE - fruit_age) / self.PERFECT_AGE
                    reward += max(0, age_score)  # Ensure non-negative
                    del self.fruits[new_pos]

                # Update position
                self.agent_pos = np.array([new_x, new_y])

        self.agent_position_history.append(tuple(self.agent_pos))

        # Age existing fruits
        fruits_to_remove = []
        for pos, fruit_info in self.fruits.items():
            fruit_info['age'] += 1.0
            if fruit_info['age'] >= self.MAX_AGE:
                fruits_to_remove.append(pos)

        # Remove expired fruits
        for pos in fruits_to_remove:
            del self.fruits[pos]

        # Spawn new fruits (random mode only)
        if self.fruit_spawning_mode == 'random':
            self._spawn_fruit_random()

        self.steps += 1

        # Check termination conditions
        if self.steps >= self.max_episode_steps:
            truncated = True

        # Early termination if all fruits eaten (strategic mode)
        if len(self.fruits) == 0 and self.fruit_spawning_mode == 'strategic':
            terminated = True

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"\n=== Step {self.steps}/{self.max_episode_steps} ===")
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    if (i, j) == tuple(self.agent_pos):
                        # Scale agent symbol: center 'A' with underscores
                        agent_symbol = "A".center(self.render_digits_len, "_")
                        print(agent_symbol, end=" ")
                    elif (i, j) in self.fruits:
                        age = int(self.fruits[(i, j)]['age'])
                        # Scale fruit age to match digit length
                        print(f"{age:{self.render_digits_len}d}", end=" ")
                    else:
                        # Scale empty space with dots
                        empty_symbol = "." * self.render_digits_len
                        print(empty_symbol, end=" ")
                print()

            print(f"Fruits: {len(self.fruits)}, Mode: {self.fruit_spawning_mode}({self.fruit_spawning_param})")
            print(f"MAX_AGE: {self.MAX_AGE}, PERFECT_AGE: {self.PERFECT_AGE}")

        elif self.render_mode == "pygame":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(self.WHITE)

            # Calculate camera position
            if self.use_camera:
                camera_x = max(0, min(self.grid_width - self.view_width,
                                      self.agent_pos[1] - self.view_width // 2))
                camera_y = max(0, min(self.grid_height - self.view_height,
                                      self.agent_pos[0] - self.view_height // 2))
            else:
                camera_x = camera_y = 0

            # Draw grid
            for i in range(self.view_height + 1):
                pygame.draw.line(self.screen, self.GRAY,
                                 (0, i * self.cell_size),
                                 (self.view_width * self.cell_size, i * self.cell_size), 2)
            for j in range(self.view_width + 1):
                pygame.draw.line(self.screen, self.GRAY,
                                 (j * self.cell_size, 0),
                                 (j * self.cell_size, self.view_height * self.cell_size), 2)

            # Draw fruits
            for pos, fruit_info in self.fruits.items():
                if self.use_camera:
                    if not (camera_x <= pos[1] < camera_x + self.view_width and
                            camera_y <= pos[0] < camera_y + self.view_height):
                        continue
                    screen_x = pos[1] - camera_x
                    screen_y = pos[0] - camera_y
                else:
                    screen_x = pos[1]
                    screen_y = pos[0]

                age = fruit_info['age']
                # Color transition: yellow -> green -> red
                if age <= self.PERFECT_AGE:
                    progress = age / self.PERFECT_AGE if self.PERFECT_AGE > 0 else 0
                    color = (int(255 * (1 - progress)), 255, 0)  # Yellow to green
                else:
                    progress = min(1.0, (age - self.PERFECT_AGE) / (self.MAX_AGE - self.PERFECT_AGE))
                    color = (int(255 * progress), int(255 * (1 - progress)), 0)  # Green to red

                fruit_rect = pygame.Rect(
                    screen_x * self.cell_size + 6,
                    screen_y * self.cell_size + 6,
                    self.cell_size - 12,
                    self.cell_size - 12
                )
                pygame.draw.rect(self.screen, color, fruit_rect)

                # Draw age number
                font = pygame.font.Font(None, 32)
                text = font.render(str(int(age)), True, self.BLACK)
                text_rect = text.get_rect(center=(
                    screen_x * self.cell_size + self.cell_size // 2,
                    screen_y * self.cell_size + self.cell_size // 2
                ))
                self.screen.blit(text, text_rect)

            # Draw agent
            if self.use_camera:
                screen_x = self.agent_pos[1] - camera_x
                screen_y = self.agent_pos[0] - camera_y
            else:
                screen_x = self.agent_pos[1]
                screen_y = self.agent_pos[0]

            if self.agent_image:
                self.screen.blit(self.agent_image,
                                 (screen_x * self.cell_size + 6,
                                  screen_y * self.cell_size + 6))
            else:
                agent_rect = pygame.Rect(
                    screen_x * self.cell_size + 6,
                    screen_y * self.cell_size + 6,
                    self.cell_size - 12,
                    self.cell_size - 12
                )
                pygame.draw.rect(self.screen, self.BLUE, agent_rect)

            # Draw information
            font = pygame.font.Font(None, 36)
            y_offset = self.view_height * self.cell_size + 10

            info_lines = [
                f"Steps: {self.steps}/{self.max_episode_steps}",
                f"Fruits: {len(self.fruits)}",
                f"Mode: {self.fruit_spawning_mode}({self.fruit_spawning_param})",
                f"Perfect Age: {self.PERFECT_AGE}, Max Age: {self.MAX_AGE}",
                f"Channels: {len(self.observation_channels)} of {len(self.CHANNEL_NAMES)}",
                f"Seed: {self.seed}"
            ]

            for i, line in enumerate(info_lines):
                text = font.render(line, True, self.BLACK)
                self.screen.blit(text, (10, y_offset + i * 30))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up resources"""
        if self.render_mode == "pygame":
            pygame.quit()


if __name__ == "__main__":
    import time

    env = AmnyamEnv(
        observation_channels=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        grid_size=(5, 60),
        max_episode_steps=80,
        fruit_spawning=('strategic', 10),
        render_mode='human',
        seed=None
        )

    for i in range(3):
        obs, info = env.reset()
        env.render()
        exit()

        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            # time.sleep(0.25)
