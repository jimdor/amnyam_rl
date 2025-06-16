from env.amnyam_env import AmnyamEnv
import time
import random


def get_action_from_key(key):
    """Convert keyboard input to action"""
    key_to_action = {
        'w': 1,  # up
        'd': 2,  # right
        's': 3,  # down
        'a': 4,  # left
        ' ': 0,  # stay
        'e': 5,  # eat
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


def main(
        render_mode: str = 'human',
        observation_channels=None,
        max_episode_steps=120,
        grid_size=(5, 60),
        fruit_spawning=('strategic', 10),
        agent_count=1,
        seed=None

):
    # Create and initialize the environment
    env = AmnyamEnv(
        render_mode=render_mode,
        observation_channels=observation_channels,
        max_episode_steps=max_episode_steps,
        grid_size=grid_size,
        fruit_spawning=fruit_spawning,
        agent_count=agent_count,
        seed=seed)
    observation, info = env.reset()

    print("\nWelcome to Amnyam!")
    print("Controls:")
    print("  w: Move up")
    print("  s: Move down")
    print("  a: Move left")
    print("  d: Move right")
    print("  space: Stay in place")
    print("  e: Eat fruit")
    print("  r: Random action")
    print("  rnd: Auto-play with random actions")
    print("  q: Quit")
    print("\nFruit stages:")
    print("  g: Growing (reward: 0)")
    print("  F: Ripe (reward: +1)")
    print("  d: Decaying (reward: -2)")
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
            key = input("Enter action (w/a/s/d/space/e/r/rnd/q): ").strip()

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
                        if action == 5:  # If eat action was taken
                            fruits_eaten += 1

                        # Print information
                        print_info(observation, reward, terminated, truncated, info, action)

                        env.render()

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
                print("Invalid input! Please use w/a/s/d/space/e/r/rnd/q")
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
            if action == 5:  # If eat action was taken
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
    main(
        observation_channels=(0, 2, 3, 4, 6, 8, 9, 10),
        grid_size=(5, 20),
        max_episode_steps=50,
        fruit_spawning=('strategic', 5),
        agent_count=1,
        render_mode='pygame',
        seed=42
    )
