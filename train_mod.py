import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env import AmnyamEnvModified
from ray import tune, air
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.connectors.env_to_module import FlattenObservations

def env_creator(env_config):
    return AmnyamEnvModified(**env_config, render_mode='silent', seed=42)

# Initialize Ray
ray.init()

# Register the environment
register_env("amnyam", env_creator)

def _env_to_module(env):
    return FlattenObservations(multi_agent=False)

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
                "head_fcnet_hiddens": [256, 128],
                "fcnet_hiddens": [512, 256],
                # "conv_filters": [
                #     (16, 2, 1, 0),
                # ],
                "fcnet_activation": "tanh",
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
        explore=True
    )
    .framework("torch")
    .debugging(seed=42)
    .training(
        train_batch_size_per_learner=1000,
        minibatch_size=100,
        lr=0.00003,
        gamma=0.99,
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
        storage_path="/Users/mihailivanov/cursor_projects/amnyam/chekhpoints",
        stop={"training_iteration": 10_000},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
            num_to_keep=1,
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
        ),
    ),
)

results = tuner.fit()