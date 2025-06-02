import gymnasium as gym
from env.amnyam_env_modified import AmnyamEnvModified
import time
import random

def get_action_from_key(key):
    """Convert keyboard input to action"""
    key_to_action = {
        'w': 0,  # up
        'd': 1,  # right
        's': 2,  # down
        'a': 3,  # left
        ' ': 4,  # stay
        'q': -1,  # quit
        'r': -2   # random action
    }
    return key_to_action.get(key.lower(), None)

def get_random_action():
    """Get a random valid action"""
    return random.randint(0, 4)

def print_info(observation, reward, terminated, truncated, info, action):
    """Print detailed information about the current state"""
    print("\n" + "="*50)
    print(f"Action taken: {action}")
    print(f"Reward: {reward}")
    print(f"Episode status: {'Terminated' if terminated else 'Truncated' if truncated else 'Continuing'}")
    if info:
        print("Additional info:", info)
    print("="*50 + "\n")

def print_episode_summary(total_reward, steps, fruits_eaten):
    """Print summary of the episode"""
    print("\n" + "="*50)
    print("EPISODE SUMMARY")
    print(f"Total Reward: {total_reward}")
    print(f"Steps Taken: {steps}")
    print(f"Fruits Eaten: {fruits_eaten}")
    print(f"Average Reward per Step: {total_reward/steps if steps > 0 else 0:.2f}")
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
    print("  r: Random action")
    print("  rnd: Auto-play with random actions")
    print("  q: Quit")
    print("\nFruit mechanics:")
    print("  - Fruits age over time")
    print("  - Perfect age is 6")
    print("  - Reward is based on how close to perfect age when eaten")
    print("  - Maximum age is 12")
    print("\nPress Enter to start...")
    input()
    
    episode_count = 0
    while True:
        # Reset episode tracking variables
        total_reward = 0
        steps = 0
        fruits_eaten = 0
        episode_count += 1
        
        print(f"\nStarting Episode {episode_count}")
        observation, info = env.reset()
        
        while True:
            # Render the current state
            env.render()
            
            # Get action from user
            key = input("Enter action (w/a/s/d/space/r/rnd/q): ").strip()
            
            # Check for autoplay mode
            if key.lower() == 'rnd':
                print("\nAutoplay mode activated! Press Ctrl+C to stop...")
                try:
                    while True:
                        action = get_random_action()
                        observation, reward, terminated, truncated, info = env.step(action)
                        
                        # Update episode tracking
                        total_reward += reward
                        steps += 1
                        if reward > 0:  # If we got a positive reward, we ate a fruit
                            fruits_eaten += 1
                        
                        # Print information
                        print_info(observation, reward, terminated, truncated, info, action)
                        
                        # Check if episode is done
                        if terminated or truncated:
                            print_episode_summary(total_reward, steps, fruits_eaten)
                            break
                        
                        time.sleep(0.5)  # Add a small delay for better visualization
                except KeyboardInterrupt:
                    print("\nAutoplay stopped by user")
                    break
            
            action = get_action_from_key(key)
            
            if action is None:
                print("Invalid input! Please use w/a/s/d/space/r/rnd/q")
                continue
            
            if action == -1:  # Quit
                env.close()
                print("\nThanks for playing!")
                return
            
            if action == -2:  # Random action
                action = get_random_action()
                print(f"Random action chosen: {action}")
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update episode tracking
            total_reward += reward
            steps += 1
            if reward > 0:  # If we got a positive reward, we ate a fruit
                fruits_eaten += 1
            
            # Print information
            print_info(observation, reward, terminated, truncated, info, action)
            
            # Check if episode is done
            if terminated or truncated:
                print_episode_summary(total_reward, steps, fruits_eaten)
                print("Press Enter to start a new episode, or 'q' to quit...")
                key = input().strip().lower()
                if key == 'q':
                    env.close()
                    print("\nThanks for playing!")
                    return
                break

if __name__ == "__main__":
    main()