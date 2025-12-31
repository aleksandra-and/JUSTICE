
from dataclasses import dataclass, field
import torch
import tyro
import yaml

from algorithms.mappo import MAPPO
from thesis_rl.envs.justice_environment import JusticeEnvironment

@dataclass
class TrainArgs:
    total_episodes: int = 100  # Total number of training episodes
    backup_interval: int = 10  # Interval for saving model checkpoints
    save_folder: str = "exp_results/runs"  # Folder to save results and models
    env_config_file: str = "thesis_rl/env_config.yaml"  # Path to environment config YAML
    num_envs: int = 1  # Number of parallel environments to use during training
    reward: str = "stepwise_marl_reward"  # Type of reward to use during training
    num_agents: int = 5  # Number of agents in the environment

@dataclass  
class EnvArgs:
    reward: str # Type of reward to use. 
    # Can also use "consumption_per_capita", "regional_temperature"
    num_agents: int # Number of agents  

  
if __name__ == "__main__":
    train_args = tyro.cli(TrainArgs)
    
    # # Load environment config from YAML
    # with open(train_args.env_config_file, "r") as f:
    #     env_config = yaml.safe_load(f)
    
    # env_args = EnvArgs(**env_config)
    env_args = EnvArgs(reward=train_args.reward, num_agents=train_args.num_agents)
    env = JusticeEnvironment(env_args)
    trainer = MAPPO(train_args)

    trainer.train_mappo(env)
    