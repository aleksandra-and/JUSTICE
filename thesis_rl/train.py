
from dataclasses import dataclass, field
import torch
import tyro
import yaml

from algorithms.mappo import MAPPO
from thesis_rl.envs.justice_environment import JusticeEnvironment
from thesis_rl.args import TrainArgs, EnvArgs

def train_mappo(train_args: TrainArgs, env_args: EnvArgs):
    env = JusticeEnvironment(env_args)
    trainer = MAPPO(train_args)

    trainer.train_mappo(env)
    

    
  
if __name__ == "__main__":
    train_args = tyro.cli(TrainArgs)
    
    # # Load environment config from YAML
    # with open(train_args.env_config_file, "r") as f:
    #     env_config = yaml.safe_load(f)
    
    # env_args = EnvArgs(**env_config)
    if train_args.algorithm.lower() == "mappo":
        env_args = EnvArgs(reward=train_args.reward, num_agents=train_args.num_agents)
        train_mappo(train_args, env_args)
    
    