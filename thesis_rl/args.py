from dataclasses import dataclass

@dataclass
class TrainArgs:
    total_episodes: int = 100  # Total number of training episodes
    backup_interval: int = 10  # Interval for saving model checkpoints
    save_folder: str = "exp_results/runs"  # Folder to save results and models
    env_config_file: str = "thesis_rl/env_config.yaml"  # Path to environment config YAML
    num_envs: int = 1  # Number of parallel environments to use during training
    reward: str = "stepwise_marl_reward"  # Type of reward to use during training
    num_agents: int = 5  # Number of agents in the environment
    algorithm: str = "mappo"  # Algorithm to use for training

@dataclass  
class EnvArgs:
    reward: str # Type of reward to use. 
    # Can also use "consumption_per_capita", "regional_temperature"
    num_agents: int # Number of agents  