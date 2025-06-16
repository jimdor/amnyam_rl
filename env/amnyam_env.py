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
        10: "current_agent_mask",  # New channel for current agent
    }

    def __init__(self,
                 render_mode=None,
                 observation_channels=(0, 1, 2, 3, 4),  # Which channels to include
                 max_episode_steps=100,
                 grid_size=(10, 10),  # (height, width) or single int for square
                 fruit_spawning=('random', 1.0),  # ('mode', temperature/group_count)
                 agent_count=1,  # New parameter for number of agents
                 seed=None):
        super().__init__()

        # Store configuration
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.observation_channels = observation_channels
        self.fruit_spawning_mode, self.fruit_spawning_param = fruit_spawning
        self.agent_count = agent_count

        # Validate agent count and strategic mode requirement
        if self.agent_count > 1 and self.fruit_spawning_mode != 'strategic':
            raise ValueError("agent_count > 1 must be used only with strategic mode")

        if self.agent_count < 1:
            raise ValueError("agent_count must be at least 1")

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

        # Initialize state for multiple agents
        self.agent_positions = None  # List of agent positions
        self.current_agent_idx = 0   # Which agent's turn it is
        self.fruits = {}  # Dictionary: (x,y) -> {'age': float}
        self.steps = 0
        self.render_mode = render_mode
        self.agent_position_histories = None  # List of deques for each agent

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
            self.view_height * self.cell_size + 250  # More space for multi-agent info
        )

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Amnyam Smart-Agile Environment (Multi-Agent)")
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)
        self.CYAN = (0, 255, 255)

        # Agent colors for multi-agent rendering
        self.AGENT_COLORS = [
            self.BLUE, self.RED, self.GREEN, self.PURPLE,
            self.ORANGE, self.CYAN, (255, 192, 203), (165, 42, 42)
        ]

        # Load amnyam image
        try:
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'amnyam.png')
            self.agent_image = pygame.image.load(image_path)
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size - 12, self.cell_size - 12))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Could not load agent image: {e}")
            self.agent_image = None

    @property
    def current_agent_pos(self):
        """Get current agent's position"""
        return self.agent_positions[self.current_agent_idx]

    def _get_observation(self):
        h, w = self.grid_height, self.grid_width
        # robust mapping: logical channel → column index in obs
        col = {ch: i for i, ch in enumerate(self.observation_channels)}

        obs = np.zeros((h, w, len(self.observation_channels)), dtype=np.float32)

        if 0 in col:
            for agent_pos in self.agent_positions:
                obs[agent_pos[0], agent_pos[1], col[0]] = 1.0

        if 10 in col:
            current_pos = self.current_agent_pos
            obs[current_pos[0], current_pos[1], col[10]] = 1.0

        inv_max_dist = 1.0 / (h + w - 2)
        pa_ratio = self.PERFECT_AGE / self.MAX_AGE

        if self.fruits:
            # keep full precision during maths
            pos = np.array(list(self.fruits.keys()), dtype=np.int16)
            ages = np.fromiter((f['age'] for f in self.fruits.values()),
                               dtype=np.float64, count=len(self.fruits))

            x, y = pos[:, 0], pos[:, 1]

            if 1 in col:
                obs[x, y, col[1]] = 1.0

            if 2 in col:
                obs[x, y, col[2]] = np.clip(ages / self.MAX_AGE, 0.0, 1.0)

            if 3 in col:
                obs[x, y, col[3]] = pa_ratio

            if 4 in col:
                current_pos = self.current_agent_pos
                d = (np.abs(x - current_pos[0]) +
                     np.abs(y - current_pos[1])) * inv_max_dist
                obs[x, y, col[4]] = d.astype(np.float32)

            if 5 in col:
                obs[x, y, col[5]] = np.clip(np.abs(ages - self.PERFECT_AGE)
                                            / self.PERFECT_AGE, 0.0, 1.0)

            if 6 in col:
                obs[x, y, col[6]] = (self.MAX_AGE - ages) / self.MAX_AGE

            if 7 in col:
                delta = self.PERFECT_AGE - ages
                ok = delta > 0
                obs[x[ok], y[ok], col[7]] = delta[ok] / self.PERFECT_AGE

        if 8 in col:
            current_history = self.agent_position_histories[self.current_agent_idx]
            for steps_ago, p in enumerate(reversed(current_history)):
                if steps_ago == 0:
                    continue
                v = 1.0 - 0.1 * steps_ago
                if v <= 0.0:
                    break
                obs[p[0], p[1], col[8]] = max(obs[p[0], p[1], col[8]], v)

        if 9 in col:
            obs[:, :, col[9]] = self.steps / self.max_episode_steps

        return obs

    def _spawn_fruit_random(self):
        """Random fruit spawning controlled by temperature"""
        if self.np_random.random() < self.spawn_probability:
            # Try to find an empty cell
            for _ in range(100):
                pos = (self.np_random.integers(0, self.grid_height),
                       self.np_random.integers(0, self.grid_width))
                # Check if position conflicts with any agent
                agent_positions_tuple = [tuple(agent_pos) for agent_pos in self.agent_positions]
                if pos not in agent_positions_tuple and pos not in self.fruits:
                    initial_age = self.np_random.uniform(0, min(3, self.PERFECT_AGE / 2))
                    self.fruits[pos] = {'age': initial_age}
                    break

    def _spawn_fruit_strategic(self):
        """Strategic fruit placement in groups"""
        self.fruits = {}

        # Distribute groups along the width
        for group in range(self.fruit_groups):
            # Calculate group center position
            group_y = (group + 1) * self.grid_width // (self.fruit_groups + 1)
            group_x = self.grid_height // 2

            # Add fruits around this center
            fruits_per_group = max(2, self.agent_count)  # Scale with agent count
            max_attempts_per_fruit = 20  # Prevent infinite loops

            for i in range(fruits_per_group):
                placed = False
                attempts = 0

                while not placed and attempts < max_attempts_per_fruit:
                    # Add positional variation
                    x = max(0, min(self.grid_height - 1,
                                   group_x + self.np_random.integers(-2, 3)))
                    y = max(0, min(self.grid_width - 1,
                                   group_y + self.np_random.integers(-2, 3)))
                    pos = (x, y)

                    # Check if position conflicts with any agent
                    agent_positions_tuple = [tuple(agent_pos) for agent_pos in self.agent_positions]
                    if pos not in agent_positions_tuple and pos not in self.fruits:
                        # Strategic aging: fruits should be ready when agents arrive
                        min_time_to_reach = min(abs(group_y - agent_pos[1]) for agent_pos in self.agent_positions)
                        age_variation = self.np_random.integers(-3, 3)
                        discount = -4
                        initial_age = max(0, self.PERFECT_AGE - min_time_to_reach + age_variation + discount)
                        self.fruits[pos] = {'age': initial_age}
                        placed = True

                    attempts += 1

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is None:
            seed = self.seed
        super().reset(seed=seed)

        # Initialize multiple agents
        self.agent_positions = []
        self.agent_position_histories = []
        self.current_agent_idx = 0

        # Agent placement strategy
        if self.fruit_spawning_mode == 'strategic':
            # Start agents at left edge with vertical spacing
            for i in range(self.agent_count):
                y_pos = max(0, min(self.grid_height - 1,
                                   i * self.grid_height // max(1, self.agent_count - 1)
                                   if self.agent_count > 1 else self.grid_height // 2))
                agent_pos = np.array([y_pos, 0])
                self.agent_positions.append(agent_pos)

                # Initialize position history for each agent
                history = deque(maxlen=10)
                history.append(tuple(agent_pos))
                self.agent_position_histories.append(history)
        else:
            # Random placement for each agent
            for i in range(self.agent_count):
                # Try to find non-overlapping positions
                placed = False
                for _ in range(100):
                    agent_pos = np.array([
                        self.np_random.integers(0, self.grid_height),
                        self.np_random.integers(0, self.grid_width)
                    ])
                    # Check if position conflicts with existing agents
                    conflict = False
                    for existing_pos in self.agent_positions:
                        if np.array_equal(agent_pos, existing_pos):
                            conflict = True
                            break

                    if not conflict:
                        self.agent_positions.append(agent_pos)
                        history = deque(maxlen=10)
                        history.append(tuple(agent_pos))
                        self.agent_position_histories.append(history)
                        placed = True
                        break

                if not placed:
                    # Fallback: place at random position even if overlapping
                    agent_pos = np.array([
                        self.np_random.integers(0, self.grid_height),
                        self.np_random.integers(0, self.grid_width)
                    ])
                    self.agent_positions.append(agent_pos)
                    history = deque(maxlen=10)
                    history.append(tuple(agent_pos))
                    self.agent_position_histories.append(history)

        # Initialize fruits based on mode
        if self.fruit_spawning_mode == 'random':
            self.fruits = {}
            # Place initial random fruits
            num_initial_fruits = self.np_random.integers(1, min(8, (self.grid_height * self.grid_width) // 10))
            for _ in range(num_initial_fruits):
                for _ in range(100):  # Try up to 100 times
                    pos = (self.np_random.integers(0, self.grid_height),
                           self.np_random.integers(0, self.grid_width))
                    agent_positions_tuple = [tuple(agent_pos) for agent_pos in self.agent_positions]
                    if pos not in agent_positions_tuple and pos not in self.fruits:
                        initial_age = self.np_random.uniform(0, min(3, self.PERFECT_AGE / 2))
                        self.fruits[pos] = {'age': initial_age}
                        break
        elif self.fruit_spawning_mode == 'strategic':
            self._spawn_fruit_strategic()

        self.steps = 0
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        """Execute one environment step"""
        reward = 0
        terminated = False
        truncated = False
        info = {'current_agent': self.current_agent_idx}

        current_pos = self.current_agent_pos

        # Handle movement for current agent
        if action == 0:  # Stay
            pass
        else:  # Movement: 1=up, 2=right, 3=down, 4=left
            directions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
            dx, dy = directions[action]

            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy

            # Check bounds
            if 0 <= new_x < self.grid_height and 0 <= new_y < self.grid_width:
                new_pos = (new_x, new_y)

                # CHECK FOR COLLISION WITH OTHER AGENTS
                collision = False
                for other_agent_idx, other_agent_pos in enumerate(self.agent_positions):
                    if (other_agent_idx != self.current_agent_idx and
                       tuple(other_agent_pos) == new_pos):
                        collision = True
                        info['collision_attempted'] = True
                        info['blocked_by_agent'] = other_agent_idx
                        break

                if not collision:
                    # Check for fruit consumption
                    if new_pos in self.fruits:
                        fruit_age = self.fruits[new_pos]['age']
                        # Reward based on proximity to perfect age
                        age_score = 1.0 - abs(self.PERFECT_AGE - fruit_age) / self.PERFECT_AGE
                        reward += max(0, age_score)  # Ensure non-negative
                        del self.fruits[new_pos]
                        info['fruit_consumed'] = True
                        info['fruit_age'] = fruit_age
                        info['age_score'] = age_score

                    # Update position (only if no collision)
                    self.agent_positions[self.current_agent_idx] = np.array([new_x, new_y])
                # If collision occurred, agent stays in current position

        # Update position history for current agent
        self.agent_position_histories[self.current_agent_idx]\
            .append(tuple(self.agent_positions[self.current_agent_idx]))

        # Switch to next agent (turn-based)
        self.current_agent_idx = (self.current_agent_idx + 1) % self.agent_count

        # Only increment steps when all agents have had their turn
        if self.current_agent_idx == 0:
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

        obs = self._get_observation()

        return obs, reward, terminated, truncated, info


    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"\n=== Step {self.steps}/{self.max_episode_steps} ===")
            print(f"Current Agent: {self.current_agent_idx + 1}/{self.agent_count}")

            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    # Check if any agent is at this position
                    agent_at_pos = None
                    for agent_idx, agent_pos in enumerate(self.agent_positions):
                        if (i, j) == tuple(agent_pos):
                            agent_at_pos = agent_idx
                            break

                    if agent_at_pos is not None:
                        # Show agent number, highlight current agent
                        if agent_at_pos == self.current_agent_idx:
                            agent_symbol = f"{agent_at_pos + 1}".center(self.render_digits_len + 1, "_")
                        else:
                            agent_symbol = f"{agent_at_pos + 1}".center(self.render_digits_len, "_")
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

            print(f"Agents: {self.agent_count}, Fruits: {len(self.fruits)},"
                  f"Mode: {self.fruit_spawning_mode}({self.fruit_spawning_param})")
            print(f"MAX_AGE: {self.MAX_AGE}, PERFECT_AGE: {self.PERFECT_AGE}")

        elif self.render_mode == "pygame":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(self.WHITE)

            # Calculate camera position based on current agent
            if self.use_camera:
                current_pos = self.current_agent_pos
                camera_x = max(0, min(self.grid_width - self.view_width,
                                      current_pos[1] - self.view_width // 2))
                camera_y = max(0, min(self.grid_height - self.view_height,
                                      current_pos[0] - self.view_height // 2))
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

            # Draw all agents
            for agent_idx, agent_pos in enumerate(self.agent_positions):
                if self.use_camera:
                    if not (camera_x <= agent_pos[1] < camera_x + self.view_width and
                            camera_y <= agent_pos[0] < camera_y + self.view_height):
                        continue
                    screen_x = agent_pos[1] - camera_x
                    screen_y = agent_pos[0] - camera_y
                else:
                    screen_x = agent_pos[1]
                    screen_y = agent_pos[0]

                # Highlight current agent with a border
                if agent_idx == self.current_agent_idx:
                    border_rect = pygame.Rect(
                        screen_x * self.cell_size + 2,
                        screen_y * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, self.BLACK, border_rect, 4)

                if self.agent_image:
                    self.screen.blit(self.agent_image,
                                     (screen_x * self.cell_size + 6,
                                      screen_y * self.cell_size + 6))

                # Draw agent number
                font = pygame.font.Font(None, 24)
                text = font.render(str(agent_idx + 1), True, self.WHITE)
                text_rect = text.get_rect(center=(
                    screen_x * self.cell_size + self.cell_size // 2,
                    screen_y * self.cell_size + self.cell_size // 2
                ))
                self.screen.blit(text, text_rect)

            # Draw information
            font = pygame.font.Font(None, 28)
            y_offset = self.view_height * self.cell_size + 10

            info_lines = [
                f"Steps: {self.steps}/{self.max_episode_steps}",
                f"Current Agent: {self.current_agent_idx + 1}/{self.agent_count}",
                f"Fruits: {len(self.fruits)}",
                f"Mode: {self.fruit_spawning_mode}({self.fruit_spawning_param})",
                f"Perfect Age: {self.PERFECT_AGE}, Max Age: {self.MAX_AGE}",
                f"Channels: {len(self.observation_channels)} of {len(self.CHANNEL_NAMES)}",
                f"Seed: {self.seed}"
            ]

            for i, line in enumerate(info_lines):
                text = font.render(line, True, self.BLACK)
                self.screen.blit(text, (10, y_offset + i * 25))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up resources"""
        if self.render_mode == "pygame":
            pygame.quit()


if __name__ == "__main__":
    import time

    env = AmnyamEnv(
        observation_channels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        grid_size=(5, 20),
        max_episode_steps=50,
        fruit_spawning=('strategic', 5),
        agent_count=3,  # Test with 3 agents
        render_mode='pygame',
        seed=42
        )

    for i in range(1):
        obs, info = env.reset()
        env.render()
        # time.sleep(1)

        terminated = truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < 40:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Agent {info['current_agent']}, Reward: {reward:.3f}")
            if 'fruit_consumed' in info:
                print(f"  -> Fruit consumed! Age: {info['fruit_age']:.1f}, Score: {info['age_score']:.3f}")

            env.render()
            time.sleep(1.0)
            step_count += 1
