"""Train an algorithm."""
from thesis_rl.args import parse_args

def main():
    """Main function."""
    args, algo_args, env_args = parse_args()

    # start training
    from harl.runners import RUNNER_REGISTRY
    import wandb
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
    
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
