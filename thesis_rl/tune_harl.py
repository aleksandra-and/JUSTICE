"""Tune an algorithm."""
import time
from thesis_rl.args import parse_args


def main():
    """Main function."""
    args, algo_args, env_args = parse_args()

    # start tuning
    from harl.runners import RUNNER_REGISTRY
    import wandb
    
    sweep_configuration = {
        "method": "bayes",
        "name": f"sweep-bayes-{args['algo']}-{env_args['reward']}",
        "metric": {"goal": "maximize", "name": "average_step_rewards"},
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
                "use_clipped_value_loss": {"distribution": "categorical", "values": [True, False]},
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
        
        args['exp_name'] = f"{args['algo']}_{env_args['num_agents']}_agents_{time.time():.0f}"
        
        # Start a new wandb run to track this script.
        wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="olaandrasz-tu-delft",
            # Set the wandb project where this run will be logged.
            project="harl_justice_exp",
            config={**algo_args['algo'], **env_args},
        )
        
        algo_args['algo'] = dict(wandb.config)
        
        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
        runner.run()
        runner.close()
            
    wandb.agent(sweep_id, train_harl, count=10)
    
    
if __name__ == "__main__":
    main()
