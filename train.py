import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.amnyam_env import AmnyamEnv
from ray import tune, air
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.connectors.env_to_module import FlattenObservations


def env_creator(env_config):
    return AmnyamEnv(**env_config,
                     observation_channels=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                     grid_size=(9, 9),
                     max_episode_steps=100,
                     fruit_spawning=('random', 1),
                     render_mode='human',
                     seed=None)


def _env_to_module(env):
    return FlattenObservations(multi_agent=False)


# Initialize Ray
ray.init()

# Register the environment
register_env("amnyam", env_creator)

# Configure the algorithm
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("amnyam", env_config={})
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config={
                "head_fcnet_hiddens": [128, 64],
                "fcnet_hiddens": [512, 256],
                # "conv_filters": [
                #     (16, 2, 1, 0),
                # ],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            }
        )
    )
    .learners(
        num_learners=0,
    )
    .env_runners(
        env_to_module_connector=_env_to_module,
        # num_env_runners=12,
        num_env_runners=0,
        num_cpus_per_env_runner=1,
        num_envs_per_env_runner=2,
        explore=True,
        rollout_fragment_length=50
    )
    .framework("torch")
    # .debugging(seed=42)
    .training(
        train_batch_size_per_learner=1000,
        minibatch_size=100,
        lr=0.00003,
        gamma=0.999,
        lambda_=0.95,
    )
)

# algo = config.build_algo()
# print(algo.get_module().encoder)
# print(algo.get_module().pi)
# exit()

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=air.RunConfig(
        storage_path="/Users/mihailivanov/code/amnyam/checkpoints",
        stop={"training_iteration": 10_000},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
            num_to_keep=1,
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
        ),
    ),
)

results = tuner.fit()
