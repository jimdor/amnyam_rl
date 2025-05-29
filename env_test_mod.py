import gymnasium as gym
from env.amnyam_env_modified import AmnyamEnvModified
import time

def get_action_from_key(key):
    """Convert keyboard input to action"""
    key_to_action = {
        'w': 0,  # up
        'd': 1,  # right
        's': 2,  # down
        'a': 3,  # left
        ' ': 4,  # stay
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
    env = AmnyamEnvModified(render_mode="pygame")
    observation, info = env.reset()
    
    print("\nWelcome to Amnyam Modified!")
    print("Controls:")
    print("  w: Move up")
    print("  s: Move down")
    print("  a: Move left")
    print("  d: Move right")
    print("  space: Stay in place")
    print("  q: Quit")
    print("\nFruit mechanics:")
    print("  - Fruits age over time")
    print("  - Perfect age is 6")
    print("  - Reward is based on how close to perfect age when eaten")
    print("  - Maximum age is 12")
    print("\nPress Enter to start...")
    input()
    
    while True:
        # Render the current state
        env.render()
        
        # Get action from user
        key = input("Enter action (w/a/s/d/space/q): ").strip()
        action = get_action_from_key(key)
        
        if action is None:
            print("Invalid input! Please use w/a/s/d/space/q")
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
            env.render()  # Render after reset
    
    env.close()
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()