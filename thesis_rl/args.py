from dataclasses import dataclass, field
import argparse
import json
from typing import List, Optional
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

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
    ensables: list
    env_name: str
    action_change: int = 1 # How much agent action can change per step
    state_type: str = 'EP'  # 'Environment Provided' or 'Function Pruned'
    num_actions: int = 21 # Number of discrete actions (e.g. 0.0, 0.05, ..., 1.0 for emissions control rate)
    # MOMARL-specific arguments
    rewards: list = field(default_factory=lambda: ['inverse_global_temperature', 'global_economic_output'])  # List of objectives for MOMARL
    weights: list = field(default_factory=lambda: [0.5, 0.5])  # Weights for linearizing multi-objective rewards
    normalize_rewards: bool = True  # Whether to normalize rewards before linearization


@dataclass
class MOMARLArgs:
    """Arguments specific to Multi-Objective MARL training."""
    weights_generation: str = "OLS"  # Method to generate weights: 'OLS' or 'uniform'
    num_weights: int = 10  # Maximum number of weight iterations
    total_uniform_weights: int = 100  # Total number of uniform weights to generate (for uniform method)
    start_uniform_weight: int = 0  # Start index for uniform weight generation
    end_uniform_weight: int = 10  # End index for uniform weight generation
    ref_point: list = field(default_factory=lambda: [0.0, 0.0])  # Reference point for hypervolume calculation
    timesteps_per_weight: int = 1000000  # Training timesteps per weight vector
    save_policies: bool = True  # Whether to save trained policies
    base_save_path: str = "results/momarl"  # Base path for saving results

from harl.utils.configs_tools import get_defaults_yaml_args, update_args

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "harl_justice",
            "harl_justice_momarl",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, harl_justice, harl_justice_momarl.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory of the trained model for evaluation.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]
        
    return args, algo_args, env_args