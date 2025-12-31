

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from thesis_rl.agents.basic_agent import Agent
from tensorboard.backend.event_processing import event_accumulator


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

LOCAL_OBSERVATIONS = [  # local observations, not shared with other agents
    "net_economic_output",
    "regional_temperature", # can remove
    "economic_damage",
    "abatement_cost",
    "emissions",
]

GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents

ALL_OBSERVATIONS = LOCAL_OBSERVATIONS + GLOBAL_OBSERVATIONS

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
    
    def plot_training_results(self, run_file=None):
        """Plot training metrics from TensorBoard events file."""
        if run_file is None:
            raise ValueError("run_file must be provided")
        
        # Construct full path
        events_file = os.path.join(self.folder, run_file)
        
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"Events file not found: {events_file}")
        
        # Parse TensorBoard events file
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        
        # Extract all scalar data
        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
        
        # Create plots
        chart_tags = [tag for tag in data.keys() if tag.startswith('charts/')]
        num_charts = len(chart_tags)
        
        fig, axes = plt.subplots(num_charts, 1, figsize=(12, 3 * num_charts))
        fig.suptitle('MAPPO Training Results - Observations Over Time', fontsize=16)
        
        # Handle single subplot case
        if num_charts == 1:
            axes = [axes]
        
        # Plot all charts
        for idx, tag in enumerate(sorted(chart_tags)):
            axes[idx].plot(data[tag]['steps'], data[tag]['values'])
            title = tag.replace('charts/', '').replace('_', ' ').title()
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Global Step')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.folder, 'training_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training results plot saved to: {output_path}")
        
        return data
        
    
    
if __name__ == "__main__":
    save_folder = "exp_results/runs/mappo_31:12:25/MAPPO_stepwise_marl_reward_100episodes_1767187416/"
    evaluator = Eval(
        args=type('obj', (object,), {
            'save_folder': save_folder
        })
    )  
    run_file="events.out.tfevents.1767187416.Lena.99972.0"
    evaluator.plot_training_results(run_file=run_file)