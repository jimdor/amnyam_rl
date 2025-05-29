import gymnasium as gym
from env.amnyam_env import AmnyamEnv
import time

def get_action_from_key(key):
    """Convert keyboard input to action"""
    key_to_action = {
        'w': 1,  # up
        'd': 2,  # right
        's': 3,  # down
        'a': 4,  # left
        ' ': 0,  # stay
        'e': 5,  # eat
        'q': -1  # quit
    }
    return key_to_action.get(key.lower(), None)

def print_info(observation, reward, terminated, truncated, info, action):
    """Print detailed information about the current state"""
    print("\n" + "="*50)
    print(f"Action taken: {action}")
    print(f"Reward: {reward}")
    print(f"Episode status: {'Terminated' if terminated else 'Truncated' if truncated else 'Continuing'}")
    if info:
        print("Additional info:", info)
    print("="*50 + "\n")

def main():
    # Create and initialize the environment
    env = AmnyamEnv(render_mode="human", grid_size=10)
    observation, info = env.reset()
    
    print("\nWelcome to Amnyam!")
    print("Controls:")
    print("  w: Move up")
    print("  s: Move down")
    print("  a: Move left")
    print("  d: Move right")
    print("  space: Stay in place")
    print("  e: Eat fruit")
    print("  q: Quit")
    print("\nFruit stages:")
    print("  g: Growing (reward: 0)")
    print("  F: Ripe (reward: +1)")
    print("  d: Decaying (reward: -2)")
    print("\nPress Enter to start...")
    input()
    
    while True:
        # Render the current state
        env.render()
        
        # Get action from user
        key = input("Enter action (w/a/s/d/space/e/q): ").strip()
        action = get_action_from_key(key)
        
        if action is None:
            print("Invalid input! Please use w/a/s/d/space/e/q")
            continue
        
        if action == -1:  # Quit
            break
        
        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print information
        print_info(observation, reward, terminated, truncated, info, action)
        
        # Check if episode is done
        if terminated or truncated:
            print("\nEpisode finished!")
            print("Press Enter to start a new episode, or 'q' to quit...")
            key = input().strip().lower()
            if key == 'q':
                break
            observation, info = env.reset()
    
    env.close()
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()