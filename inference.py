import os
import tree
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.amnyam_env import AmnyamEnv
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.connectors.env_to_module import EnvToModulePipeline, FlattenObservations
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
import time

def env_creator(env_config):
    return AmnyamEnv(**env_config)

def _env_to_module(env):
    return FlattenObservations(multi_agent=False)

def main():
    # Initialize Ray
    ray.init()

    # Register the environment
    register_env("amnyam", env_creator)

    # Configure the algorithm (exactly as in training)
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("amnyam", env_config={"grid_size": 10,})
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": [256, 128],
                    "fcnet_hiddens": [1024, 512],
                    # "conv_filters": [
                    #     (16, 2, 1, 0),
                    # ],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": True,
                }
            )
        )
        .training(
            train_batch_size_per_learner=1000,
            minibatch_size=100,
            lr=0.00003,
            gamma=0.99,
            lambda_=0.95,
        )
        .learners(
            num_learners=0
        )
        .env_runners(
            env_to_module_connector=_env_to_module,
            num_env_runners=0,
            explore=True
        )
        .framework("torch")
    )

    # Build the algorithm using the new API
    algo = config.build_algo()

    # Get the absolute path for the checkpoint
    checkpoint_path = os.path.abspath("chekhpoints/PPO_2025-06-02_14-25-31/PPO_amnyam_4aeaa_00000_0_2025-06-02_14-25-31/checkpoint_000004")
    print(f"Loading checkpoint from: {checkpoint_path}")
    algo.restore(checkpoint_path)

    print("Creating inference pipeline...")

    # Create environment
    print("Creating environment...", end="")
    env = env_creator({"grid_size": 10, "render_mode": "pygame", "seed": 444442})
    print(" ok")

    # Create the env-to-module pipeline from the checkpoint
    print("Restoring env-to-module connector...", end="")
    env_to_module = EnvToModulePipeline.from_checkpoint(
        os.path.join(
            checkpoint_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_ENV_TO_MODULE_CONNECTOR,
        )
    )
    print(" ok")

    # Create RLModule from checkpoint
    print("Restoring RLModule...", end="")
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            DEFAULT_MODULE_ID,
        )
    )
    print(" ok")

    # Create module-to-env pipeline
    print("Restoring module-to-env connector...", end="")
    module_to_env = ModuleToEnvPipeline.from_checkpoint(
        os.path.join(
            checkpoint_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_MODULE_TO_ENV_CONNECTOR,
        )
    )
    print(" ok")

    print("\nStarting inference with trained agent!")
    print("Press Ctrl+C to stop...")

    try:
        # Initialize episode
        obs, _ = env.reset()
        episode = SingleAgentEpisode(
            observations=[obs],
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        while True:
            # Process observation through env-to-module pipeline
            shared_data = {}
            input_dict = env_to_module(
                episodes=[episode],
                rl_module=rl_module,
                explore=False,  # No exploration during inference
                shared_data=shared_data,
            )

            # Get action from RLModule
            rl_module_out = rl_module.forward_inference(input_dict)

            # Process action through module-to-env pipeline
            to_env = module_to_env(
                batch=rl_module_out,
                episodes=[episode],
                rl_module=rl_module,
                explore=False,
                shared_data=shared_data,
            )

            # Get action and step environment
            action = to_env.pop(Columns.ACTIONS)[0]
            env.render()
            obs, reward, terminated, truncated, info = env.step(action)

            # Update episode
            episode.add_env_step(
                obs,
                action,
                reward,
                terminated=terminated,
                truncated=truncated,
                extra_model_outputs={k: v[0] for k, v in to_env.items()},
            )

            # Add a small delay to make the visualization more readable
            time.sleep(0.1)

            # Check if episode is done
            if episode.is_done:
                print(f"\nEpisode finished! Total reward = {episode.get_return()}")
                print("Starting new episode...")
                obs, _ = env.reset()
                episode = SingleAgentEpisode(
                    observations=[obs],
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )
                time.sleep(1)  # Pause before starting new episode

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        env.close()
        ray.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main() 