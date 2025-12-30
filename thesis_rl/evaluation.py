

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from thesis_rl.agents.basic_agent import Agent


def batchify_obs(obs):
    """Converts PZ observations to batch tensor."""
    return torch.tensor(np.stack([obs[a] for a in obs], axis=0), dtype=torch.float32)


def batchify(x):
    """Converts PZ dict to batch tensor."""
    return torch.tensor(np.stack([x[a] for a in x], axis=0).flatten(), dtype=torch.float32)


def unbatchify(x, env):
    """Converts tensor to PZ dict."""
    x = x.numpy()
    return {a: x[i] for i, a in enumerate(env.possible_agents)}

class Eval:
    
    def __init__(self, args):
        self.folder = args.save_folder
    
    def evalaute_policy(self, env, agent=None, weights_file=None, render_env=True, eval_episodes=3):
        # Environment variables
        num_agents = len(env.possible_agents)
        num_actions = env.action_space(env.possible_agents[0]).n
        obs_size = env.observation_space(env.possible_agents[0]).shape[0]
        

        print(f"Evaluating Policy")
        print(f"Agents: {num_agents}, Actions: {num_actions}, Obs size: {obs_size}\n")

        if weights_file is None and agent is None:
            raise ValueError("No trained model found for evaluation.")
        
        """ AGENT SETUP """
        if weights_file is not None:
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"Weights file not found: {weights_file}")
            agent = Agent(obs_size, num_actions)
            agent.load_state_dict(torch.load(weights_file))

        """ EVALUATION LOGIC """
        for episode in range(eval_episodes):
            next_obs, info = env.reset(seed=None)
            total_episodic_return = np.zeros(num_agents)

            for step in range(env.num_years):
                obs = batchify_obs(next_obs)

                with torch.no_grad():
                    actions, _, _, _ = agent.get_action_and_value(obs)

                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                total_episodic_return += batchify(rewards).cpu().numpy()

                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

            if episode == 0 and render_env:
                # render first episode
                env.render(f"{self.folder}/final_evaluation.png")
        
        
        print(f"Evaluation over {eval_episodes} episodes complete.")
        print(f"Avg Episodic Return: {np.mean(total_episodic_return, axis=0):.2f}")
        return total_episodic_return
    
    def plot_training_results(self, run_name=None):
        pass