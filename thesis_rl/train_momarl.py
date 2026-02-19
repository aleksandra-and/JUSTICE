"""
Multi-Objective Multi-Agent Reinforcement Learning (MOMARL) Training with HARL.

Extends HARL training to support multi-objective optimization using MOMAland.
Uses Optimistic Linear Support (OLS) or uniform weight generation to train
a set of policies covering the Pareto front.
"""

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict

import wandb
from harl.runners import RUNNER_REGISTRY
from thesis_rl.args import parse_args, MOMARLArgs

# MORL imports
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
import morl_baselines.common.weights


def parse_momarl_args():
    """Parse MOMARL-specific arguments in addition to standard HARL args."""
    import argparse
    
    # First get standard HARL args
    args, algo_args, env_args = parse_args()
    
    # Add MOMARL-specific argument parsing
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weights_generation", type=str, default="OLS",
                        choices=["OLS", "uniform"],
                        help="Weight generation method: OLS or uniform")
    parser.add_argument("--num_weights", type=int, default=10,
                        help="Maximum number of weight iterations")
    parser.add_argument("--start_uniform_weight", type=int, default=0,
                        help="Start index for uniform weights")
    parser.add_argument("--end_uniform_weight", type=int, default=10,
                        help="End index for uniform weights")
    parser.add_argument("--ref_point", type=float, nargs='+', default=[0.0, 0.0],
                        help="Reference point for hypervolume calculation")
    parser.add_argument("--timesteps_per_weight", type=int, default=1000000,
                        help="Training timesteps per weight vector")
    parser.add_argument("--save_policies", action="store_true", default=True,
                        help="Whether to save trained policies")
    parser.add_argument("--base_save_path", type=str, default="results/momarl",
                        help="Base path for saving results")
    
    momarl_args, _ = parser.parse_known_args()
    momarl_args = vars(momarl_args)
    
    return args, algo_args, env_args, momarl_args


def evaluate_policy_vector_rewards(runner, env_args, num_episodes=5):
    """
    Evaluate a trained policy and return vector (non-linearized) rewards.
    
    Args:
        runner: Trained HARL runner
        env_args: Environment arguments
        num_episodes: Number of evaluation episodes
        
    Returns:
        mean_vec_return: Mean vector return across episodes (shape: num_objectives)
    """
    import torch
    from thesis_rl.envs.justice_environment_moma import JusticeEnvironmentMOMA
    from thesis_rl.args import EnvArgs
    
    # Create unwrapped MOMA environment for evaluation
    eval_env_args = EnvArgs(**{k: v for k, v in env_args.items() 
                               if k in EnvArgs.__dataclass_fields__})
    eval_env = JusticeEnvironmentMOMA(eval_env_args)
    
    num_objectives = eval_env.num_objectives
    vec_returns = []
    
    # Get device from runner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for episode in range(num_episodes):
        obs, infos = eval_env.reset(seed=episode)
        episode_vec_reward = np.zeros(num_objectives)
        done = False
        
        # Initialize RNN states for all agents
        rnn_states = {}
        for i, agent in enumerate(eval_env.agents):
            # Create dummy RNN states (shape: 1, recurrent_N, hidden_size)
            rnn_states[agent] = np.zeros((1, 1, 64), dtype=np.float32)
        
        while not done:
            # Get actions from trained policy
            actions = {}
            for i, agent in enumerate(eval_env.agents):
                obs_array = np.expand_dims(obs[agent], axis=0)  # (1, obs_dim)
                masks = np.ones((1, 1), dtype=np.float32)
                avail_actions = np.array([infos[agent].get('action_mask', None)])
                
                with torch.no_grad():
                    action, _, rnn_state = runner.actor[i].get_actions(
                        obs_array,
                        rnn_states[agent],
                        masks,
                        avail_actions if avail_actions[0] is not None else None,
                    )
                
                rnn_states[agent] = rnn_state.cpu().numpy() if hasattr(rnn_state, 'cpu') else rnn_state
                action = action.cpu().numpy() if hasattr(action, 'cpu') else action
                actions[agent] = int(action.flatten()[0])
            
            obs, rewards, terminated, truncated, infos = eval_env.step(actions)
            
            # Accumulate vector rewards (rewards are now vectors)
            for agent in rewards:
                episode_vec_reward += rewards[agent]
                break  # All agents get same global objectives
            
            done = all(terminated.values()) or all(truncated.values())
        
        vec_returns.append(episode_vec_reward)
    
    mean_vec_return = np.mean(vec_returns, axis=0)
    return mean_vec_return


def train_single_weight(args, algo_args, env_args, weights, weight_idx, wandb_run=None):
    """
    Train a single policy for a given weight vector.
        
    Returns:
        runner: Trained HARL runner
    """
    # Update env_args with current weights
    env_args['weights'] = weights.tolist()
    
    # Create runner and train
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    
    return runner


def main():
    """Main MOMARL training function."""
    args, algo_args, env_args, momarl_args = parse_momarl_args()
    
    # Ensure we're using the MOMARL environment
    if args["env"] != "harl_justice_momarl":
        print(f"Warning: Switching environment from {args['env']} to harl_justice_momarl")
        args["env"] = "harl_justice_momarl"
    
    # Get number of objectives from env config
    num_objectives = len(env_args.get('rewards', ['inverse_global_temperature', 'global_economic_output']))
    
    print(f"=== MOMARL Training ===")
    print(f"Algorithm: {args['algo']}")
    print(f"Objectives: {env_args.get('rewards', ['inverse_global_temperature', 'global_economic_output'])}")
    print(f"Weights generation: {momarl_args['weights_generation']}")
    print(f"Max weights: {momarl_args['num_weights']}")
    
    # Create experiment group name for wandb
    timestamp = int(time.time())
    exp_name = f"momarl_{args['algo']}_{momarl_args['weights_generation']}_{timestamp}"
    wandb_group = exp_name  # All runs in this experiment share this group
    args['exp_name'] = exp_name
    
    # Setup results storage
    results_path = Path(momarl_args['base_save_path']) / exp_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    eval_results = {
        "weights": [],
        "vector_returns": [],
        "objectives": env_args.get('rewards', ['inverse_global_temperature', 'global_economic_output']),
    }
    
    # Initialize weight generation
    use_ols = momarl_args['weights_generation'] == "OLS"
    ols = None
    all_weights = None
    
    if use_ols:
        ols = LinearSupport(num_objectives=num_objectives, epsilon=0.0, verbose=True)
        weights = ols.next_weight()
       
    
    if not use_ols:
        print(f"Using uniform weight generation")
        
        # For distributed training: generate total_uniform_weights, then slice
        all_weights = morl_baselines.common.weights.equally_spaced_weights(
            num_objectives, momarl_args['num_weights'],
        )
        all_weights = all_weights[momarl_args['start_uniform_weight']:momarl_args['end_uniform_weight']]
        
        momarl_args['num_weights'] = len(all_weights)
        weights = all_weights[0]
        print(f"Generated {len(all_weights)} weights from {all_weights[0]} to {all_weights[-1]}")
    
    weight_idx = 0
    
    # Main training loop over weights
    while weight_idx < momarl_args['num_weights']:
        if use_ols and ols is not None and ols.ended():
            print("OLS terminated - Pareto front covered")
            break
        
        print(f"\n{'='*60}")
        print(f"Weight iteration {weight_idx + 1}/{momarl_args['num_weights']}")
        print(f"Current weights: {weights}")
        print(f"{'='*60}")
        
        # Update experiment name for this weight
        weight_run_name = f"weight_{weight_idx}_[{weights[0]:.2f},{weights[1]:.2f}]"
        args['exp_name'] = f"{exp_name}_w{weight_idx}"
        
        # Start a new wandb run for this weight (grouped with other weights)
        wandb.init(
            entity="olaandrasz-tu-delft",
            project="harl_justice_momarl",
            group=wandb_group,  # Groups all weight runs together
            name=weight_run_name,
            config={
                "weight_idx": weight_idx,
                "weights": weights.tolist(),
                **algo_args.get('algo', {}),
                **env_args,
                **momarl_args,
            },
        )
        
        # Train policy for current weights
        runner = train_single_weight(args, algo_args, env_args, weights, weight_idx)
        
        # Evaluate policy to get vector returns
        try:
            vec_return = evaluate_policy_vector_rewards(runner, env_args, num_episodes=5)
            print(f"Vector return: {vec_return}")
        except Exception as e:
            print(f"Warning: Could not evaluate policy: {e}")
            # Use placeholder if evaluation fails
            vec_return = np.zeros(num_objectives)
        
        # Store results
        eval_results["weights"].append(weights.tolist())
        eval_results["vector_returns"].append(vec_return.tolist())
        
        # Log final evaluation to this weight's run
        wandb.log({
            "final/vector_return": vec_return.tolist(),
            **{f"final/return_obj_{i}": vec_return[i] for i in range(num_objectives)},
        })
        
        # Log summary metrics for this weight
        wandb.summary["weight_idx"] = weight_idx
        wandb.summary["weights"] = weights.tolist()
        wandb.summary["vector_return"] = vec_return.tolist()
        for i in range(num_objectives):
            wandb.summary[f"return_obj_{i}"] = vec_return[i]
        
        # Save intermediate results
        with open(results_path / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Close runner and finish this weight's wandb run
        runner.close()
        wandb.finish()
        
        # Update weights for next iteration
        if use_ols and ols is not None:
            ols.add_solution(vec_return, weights)
            if not ols.ended():
                weights = ols.next_weight()
        
        if not use_ols and all_weights is not None:
            weight_idx += 1
            if weight_idx < len(all_weights):
                weights = all_weights[weight_idx]
        elif use_ols:
            weight_idx += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("MOMARL Training Complete!")
    print(f"Trained {weight_idx} policies")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")
    
    # Log final Pareto front approximation
    if len(eval_results["vector_returns"]) > 0:
        vec_returns = np.array(eval_results["vector_returns"])
        print(f"\nPareto front approximation:")
        for i, (w, vr) in enumerate(zip(eval_results["weights"], eval_results["vector_returns"])):
            print(f"  Policy {i}: weights={w}, return={vr}")
        
        # Create a summary wandb run for the Pareto front
        wandb.init(
            entity="olaandrasz-tu-delft",
            project="harl_justice_momarl",
            group=wandb_group,
            name=f"summary_pareto_front",
            job_type="summary",
            config={
                "num_policies": len(eval_results["weights"]),
                "objectives": eval_results["objectives"],
                **momarl_args,
            },
        )
        
        # Create a table for the Pareto front
        objectives = eval_results.get("objectives", ["obj_0", "obj_1"])
        columns = ["policy_idx", "weight_0", "weight_1"] + [f"return_{obj}" for obj in objectives]
        pareto_table = wandb.Table(columns=columns)
        
        for i, (w, vr) in enumerate(zip(eval_results["weights"], eval_results["vector_returns"])):
            row = [i, w[0], w[1]] + vr
            pareto_table.add_data(*row)
        
        wandb.log({"pareto_front": pareto_table})
        
        # Log scatter plot of Pareto front
        if len(objectives) >= 2:
            data = [[vr[0], vr[1]] for vr in eval_results["vector_returns"]]
            table = wandb.Table(data=data, columns=[objectives[0], objectives[1]])
            wandb.log({
                "pareto_scatter": wandb.plot.scatter(
                    table, objectives[0], objectives[1],
                    title="Pareto Front Approximation"
                )
            })
        
        wandb.finish()
    
    # Save final results
    with open(results_path / "final_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    


if __name__ == "__main__":
    main()
