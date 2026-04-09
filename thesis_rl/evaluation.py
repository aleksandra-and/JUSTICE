"""Evaluate an algorithm."""
from thesis_rl.args import parse_args

def main():
    """Main function."""
    args, algo_args, env_args = parse_args()
    # start evaluation
    from harl.runners import RUNNER_REGISTRY
    
    algo_args["render"]["use_render"] = True
    algo_args["train"]["model_dir"] = args["model_dir"]
    
    env_args["evaluation_output_dir"] = args["evaluation_output_dir"]
    
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
