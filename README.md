# Custom Gymnasium Environment

This is a basic template for a custom Gymnasium environment. The environment is currently a blank slate that you can customize according to your needs.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The environment is defined in `custom_env.py`. You can import and use it like this:
```python
from custom_env import CustomEnv

env = CustomEnv()
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

## Customization

The current environment includes:
- A discrete action space with 4 actions (up, down, left, right)
- An observation space of 64x64 RGB images
- Basic step and reset methods that you can customize

To customize the environment:
1. Modify the action and observation spaces in `__init__`
2. Implement your environment logic in the `step` method
3. Add any necessary rendering code in the `render` method
4. Add cleanup code in the `close` method if needed 