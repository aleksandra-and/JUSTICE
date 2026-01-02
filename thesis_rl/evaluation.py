

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from thesis_rl.agents.basic_agent import Agent
from tensorboard.backend.event_processing import event_accumulator

from thesis_rl.args import EnvArgs
from thesis_rl.envs.justice_environment import JusticeEnvironment


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
    
    def evalaute_policy(self, env, agent=None, weights_file=None, render_env=True, eval_episodes=5):
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
            if not os.path.exists(os.path.join(self.folder, weights_file)):
                raise FileNotFoundError(f"Weights file not found: {weights_file}")
            agent = Agent(obs_size, num_actions)
            agent.load_state_dict(torch.load(os.path.join(self.folder, weights_file), weights_only=True))

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
                print("Rendering first evaluation episode...")
                env.render(f"{self.folder}/final_evaluation.png")
        
        
        print(f"Evaluation over {eval_episodes} episodes complete.")
        print(f"Avg Episodic Return: {np.mean(total_episodic_return, axis=0):.2f}")
        return total_episodic_return
    
    
    def plot_rewards_learning_curve(self, logs_folder=None, output_name='learning_curve.png', smooth_window=10):
        """Plot the learning curve (episode rewards over training)."""
        if logs_folder is None:
            # Default to logs subfolder
            logs_folder = os.path.join(self.folder, 'logs')
        
        if not os.path.exists(logs_folder):
            raise FileNotFoundError(f"Logs folder not found: {logs_folder}")
        
        # Look for rewards event file
        rewards_path = os.path.join(logs_folder, 'train_episode_rewards/aver_rewards/')
        
        if not os.path.exists(rewards_path):
            raise FileNotFoundError(f"Rewards path not found: {rewards_path}")
        
        # Find the events file in the rewards directory
        event_files = [f for f in os.listdir(rewards_path) if f.startswith('events.out.tfevents')]
        
        if not event_files:
            raise FileNotFoundError(f"No events file found in {rewards_path}")
        
        events_file = os.path.join(rewards_path, event_files[0])
        print(f"Reading from: {events_file}")
        
        # Parse TensorBoard events file
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        
        # Extract rewards data
        tags = ea.Tags()['scalars']
        print(f"Available tags: {tags}")
        
        if not tags:
            raise ValueError("No scalar tags found in the events file!")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            print(f"Tag '{tag}': {len(events)} data points")
            print(f"  Steps range: {min(steps) if steps else 'N/A'} to {max(steps) if steps else 'N/A'}")
            print(f"  Values range: {min(values) if values else 'N/A'} to {max(values) if values else 'N/A'}")
            
            all_data[tag] = {'steps': steps, 'values': values}
            
            ax.plot(steps, values, linewidth=1, color='lightblue')
        
        ax.set_title('Learning Curve - Episode Rewards Over Training', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Episode Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.folder, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nLearning curve saved to: {output_path}")
        
        return all_data
    
    def plot_agent_training_info(self, logs_folder=None, output_name='agent_training_metrics.png', smooth_window=10):
        """
        Plot agent-specific training metrics (policy loss, entropy, ratio, actor grad norm).
        """
        if logs_folder is None:
            # Default to logs subfolder
            logs_folder = os.path.join(self.folder, 'logs')
        
        if not os.path.exists(logs_folder):
            raise FileNotFoundError(f"Logs folder not found: {logs_folder}")
        
        # Metrics to plot for each agent
        metrics = ['policy_loss', 'dist_entropy', 'ratio', 'actor_grad_norm']
        
        # Find all agent folders
        agent_folders = sorted([d for d in os.listdir(logs_folder) if d.startswith('agent')])
        
        if not agent_folders:
            raise FileNotFoundError(f"No agent folders found in {logs_folder}")
        
        print(f"Found {len(agent_folders)} agents: {agent_folders}")
        
        # Create subplots - one row per metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_folders)))
        
        all_data = {}
        total_points = 0
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            for agent_idx, agent_folder in enumerate(agent_folders):
                # Path structure: logs/agent0/policy_loss/agent0/policy_loss/events.out.tfevents...
                metric_path = os.path.join(logs_folder, agent_folder, metric, agent_folder, metric)
                
                if not os.path.exists(metric_path):
                    print(f"Warning: {metric_path} not found, skipping")
                    continue
                
                # Find events file
                event_files = [f for f in os.listdir(metric_path) if f.startswith('events.out.tfevents')]
                
                if not event_files:
                    print(f"Warning: No events file in {metric_path}, skipping")
                    continue
                
                events_file = os.path.join(metric_path, event_files[0])
                
                # Parse TensorBoard events file
                ea = event_accumulator.EventAccumulator(events_file)
                ea.Reload()
                
                tags = ea.Tags()['scalars']
                
                if not tags:
                    continue
                
                # Usually there's only one tag per metric file
                for tag in tags:
                    events = ea.Scalars(tag)
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    
                    total_points += len(steps)
                    
                    if agent_folder not in all_data:
                        all_data[agent_folder] = {}
                    all_data[agent_folder][metric] = {'steps': steps, 'values': values}
                
                    
                    if len(steps) > 0:
                        # Plot raw data with transparency
                        ax.plot(steps, values, alpha=0.4, linewidth=1, color=colors[agent_idx])
                        
                        # Plot smoothed data if enough points
                        if len(values) > smooth_window:
                            smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                            smoothed_steps = steps[smooth_window-1:]
                            ax.plot(smoothed_steps, smoothed, label=agent_folder, linewidth=2, color=colors[agent_idx])
                        else:
                            ax.plot(steps, values, label=agent_folder, linewidth=2, marker='o', markersize=8, color=colors[agent_idx])
            
            # Format subplot
            metric_title = metric.replace('_', ' ').title()
            ax.set_title(f'{metric_title} - All Agents', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        
        plt.suptitle('Agent Training Metrics', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.folder, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nAgent training metrics saved to: {output_path}")
        
        return all_data
    
    def plot_training_results(self):
        """Plot training metrics from TensorBoard events file.""" 
        # Construct full path
        events_file = [f for f in os.listdir(self.folder) if f.startswith('events.out.tfevents')][0]
        events_file = os.path.join(self.folder, events_file)
        
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"Events file not found: {events_file}")
        
        # Parse TensorBoard events file
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        
        print(f"\nPlotting training results...")
        # Extract all scalar data
        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
        
        # Plot mean episodic reward as separate learning curve
        if 'mean_episodic_return' in data:
            reward_tag = 'mean_episodic_return'
            
            fig_reward, ax_reward = plt.subplots(figsize=(12, 7))
            
            steps = data[reward_tag]['steps']
            values = data[reward_tag]['values']
            
            # Plot raw data with transparency
            ax_reward.plot(steps, values, linewidth=1, color='steelblue', label='Raw Returns')
            
            
            ax_reward.set_title('Learning Curve - Mean Episodic Return', fontsize=18, fontweight='bold', pad=20)
            ax_reward.set_xlabel('Training Steps', fontsize=14)
            ax_reward.set_ylabel('Mean Episodic Return', fontsize=14)
            ax_reward.grid(True, alpha=0.3, linestyle='--')
            ax_reward.legend(fontsize=12, loc='best')
            
            plt.tight_layout()
            
            # Save learning curve
            learning_curve_path = os.path.join(self.folder, 'learning_curve.png')
            plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {learning_curve_path}")
            plt.close(fig_reward)
        
        # Create plots for other charts
        chart_tags = [tag for tag in data.keys() if tag.startswith('charts/')]
        num_charts = len(chart_tags)
        
        if num_charts > 0:
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
            plt.close(fig)
        
        return data
        
    
    
if __name__ == "__main__":
    save_folder = "results/harl_justice/harl_justice/happo/installtest/seed-00001-2026-01-02-18-40-27"
    evaluator = Eval(
        args=type('obj', (object,), {
            'save_folder': save_folder
        })
    )  
    
    # env = JusticeEnvironment(EnvArgs(reward='global_temperature', num_agents=5))
    # evaluator.evalaute_policy(env=env, weights_file='final_mappo_agent.pt', render_env=True)
    # evaluator.plot_training_results()
    evaluator.plot_rewards_learning_curve()
    evaluator.plot_agent_training_info()
    