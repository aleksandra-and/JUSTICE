"""Train an algorithm."""
import argparse
import json
import random
import time
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
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
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
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

    # start training
    from harl.runners import RUNNER_REGISTRY
    import wandb
    
    sweep_configuration = {
        "method": "bayes",
        "name": f"sweep-bayes-{time.time():.0f}",
        "metric": {"goal": "maximize", "name": "aver_episode_rewards"},
        "parameters": { 
            # "model" args
                #"lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-3},
                #"critic_lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-3},
            # "algo" args
                "gamma": {"distribution": "uniform", "min": 0.9, "max": 0.999},
                "gae_lambda": {"distribution": "uniform", "min": 0.8, "max": 1.0},
                #"num_minibatches": {"distribution": "categorical", "values": [32, 50, 64]},
                "ppo_epochs": {"distribution": "int_uniform", "min": 3, "max": 10},
                "use_gae": {"distribution": "categorical", "values": [True, False]},
                "clip_param": {"distribution": "uniform", "min": 0.1, "max": 0.3},
                "use_clipped_value_los": {"distribution": "categorical", "values": [True, False]},
                "entropy_coef": {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.01},
                "value_loss_coef": {"distribution": "uniform", "min": 0.1, "max": 1.0},
                "max_grad_norm": {"distribution": "uniform", "min": 0.3, "max": 1.0},
            #"target_kl": {"distribution": "uniform", "min": 0.01, "max": 0.3},
        }
    }
    
    sweep_id = wandb.sweep(sweep_configuration, project="harl_justice_exp")
    
    def train_harl():
        # start training
        import time
        
        print(f"Tracking experiment with wandb")
        
        args['exp_name'] = f"{args['algo']}_{args['env']}_{time.time():.0f}"
        
        # Start a new wandb run to track this script.
        wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="olaandrasz-tu-delft",
            # Set the wandb project where this run will be logged.
            project="harl_justice_exp",
            name=args['exp_name'],
            config={**algo_args['algo'], **env_args},
        )
        
        algo_args['algo'] = dict(wandb.config)
        
        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
        runner.run()
        runner.close()
            
    wandb.agent(sweep_id, train_harl, count=10)
    
    
if __name__ == "__main__":
    main()
