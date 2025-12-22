
from dataclasses import dataclass, field
import tyro
import yaml

from algorithms.mappo import MAPPO
from envs.justice_environment import JusticeEnvironment

@dataclass
class TrainArgs:
    total_episodes: int = 200  # Total number of training episodes
    backup_interval: int = 50  # Interval for saving model checkpoints
    save_folder: str = "exp_results/runs"  # Folder to save results and models
    env_config_file: str = "thesis-rl/env_config.yaml"  # Path to environment config YAML

@dataclass  
class EnvArgs:
    reward: str = "stepwise_marl_reward" # Type of reward to use. 
    # Can also use "consumption_per_capita", "regional_temperature"
    num_agents: int = 5 # Number of agents  

  
if __name__ == "__main__":
    train_args = tyro.cli(TrainArgs)
    
    # Load environment config from YAML
    with open(train_args.env_config_file, "r") as f:
        env_config = yaml.safe_load(f)
    
    env_args = EnvArgs(**env_config)
    env = JusticeEnvironment(env_args)
    trainer = MAPPO(train_args)

    trainer.train_mappo(env)
    trainer.evaluate_mappo(env)