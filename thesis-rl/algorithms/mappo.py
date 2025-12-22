

import torch
import time
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch.optim as optim

from agents.basic_agent import Agent    

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

class MAPPO:
   
    def __init__(self, args):
       # Hyperparameters
        self.device = torch.device("cpu")
        
        # Learning hyperparameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.minibatch_size = 128
        
        self.agent = None
        
        self.total_episodes = args.total_episodes
        self.print_interval = args.backup_interval
        
        timestamp = datetime.now().strftime("%d:%m:%y_%H:%M")
        self.folder = args.save_folder + f"/mappo_{timestamp}"
        if not os.path.exists(f"{self.folder}/checkpoints"):
            os.makedirs(f"{self.folder}/checkpoints")
    
    def train_mappo(self, env):
        # Environment variables
        num_agents = len(env.possible_agents)
        num_actions = env.action_space(env.possible_agents[0]).n
        obs_size = env.observation_space(env.possible_agents[0]).shape[0]

        print(f"Agents: {num_agents}, Actions: {num_actions}, Obs size: {obs_size}\n")

        """ LEARNER SETUP """
        self.agent = Agent(num_actions=num_actions, obs_size=obs_size).to(self.device)
        optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        max_cycles = env.num_steps

        """ ALGO LOGIC: EPISODE STORAGE"""
        end_step = 0
        total_episodic_return = 0
        rb_obs = torch.zeros((max_cycles, num_agents, obs_size)).to(self.device)
        rb_actions = torch.zeros((max_cycles, num_agents)).to(self.device)
        rb_logprobs = torch.zeros((max_cycles, num_agents)).to(self.device)
        rb_rewards = torch.zeros((max_cycles, num_agents)).to(self.device)
        rb_terms = torch.zeros((max_cycles, num_agents)).to(self.device)
        rb_values = torch.zeros((max_cycles, num_agents)).to(self.device)

        """ TRAINING LOGIC """
        # train for n number of episodes
        start_time = time.time()
        returns_history = np.zeros((self.total_episodes, num_agents))
        for episode in range(self.total_episodes):
            # reset the episodic return
            total_episodic_return = np.zeros(num_agents)
            
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = env.reset(seed=None)

                # each episode has num_steps
                for step in range(0, max_cycles):
                    # rollover the observation
                    obs = batchify_obs(next_obs)

                    # get action from the agent
                    actions, logprobs, _, values = self.agent.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(
                        unbatchify(actions, env)
                    )

                    # add to episode storage
                    rb_obs[step] = obs
                    rb_rewards[step] = batchify(rewards)
                    rb_terms[step] = batchify(terms)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()

                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break
                
            returns_history[episode] = total_episodic_return
            # bootstrap value if not done
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(self.device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t]
                        + self.gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                    )
                    rb_advantages[t] = delta + self.gamma * self.gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values

            # convert our episodes to batch of individual transitions
            b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
            b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
            b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
            b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(10):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), self.minibatch_size):
                    # select the indices we want to train on
                    end = start + self.minibatch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = self.agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            if episode % self.print_interval == 0:
                print(f"Training episode {episode}/{self.total_episodes}")
                print(f"    Avg Episodic Return: {np.mean(total_episodic_return):.2f}")
                print(f"    Avg Episode Time: {(time.time() - start_time)/(episode+1):.2f} seconds")
                print("")
                print(f"    Value Loss: {v_loss.item()}")
                print(f"    Policy Loss: {pg_loss.item()}")
                print(f"    Old Approx KL: {old_approx_kl.item()}")
                print(f"    Approx KL: {approx_kl.item()}")
                print(f"    Clip Fraction: {np.mean(clip_fracs)}")
                print(f"    Explained Variance: {explained_var.item()}")
                print("\n-------------------------------------------")
            # Save checkpoint
            if episode % self.print_interval == 0 and episode > 0:
                torch.save({
                    'episode': episode,
                    'return': total_episodic_return,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{self.folder}/checkpoints/checkpoint_ep{episode}.pt")

        # Final save
        torch.save(self.agent.state_dict(), f"{self.folder}/justice_ppo_final.pt")
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(returns_history.sum(axis=1))
        plt.xlabel('Episode')
        plt.ylabel('Total Return')
        plt.title('Learning Curve')
        plt.grid(True)
        plt.savefig(f"{self.folder}/learning_curve.png")
        print("Learning curve saved")
    
    def evaluate_mappo(self, env, eval_episodes=3):
        # Environment variables
        num_agents = len(env.possible_agents)
        num_actions = env.action_space(env.possible_agents[0]).n
        obs_size = env.observation_space(env.possible_agents[0]).shape[0]

        print(f"Evaluating MAPPO")
        print(f"Agents: {num_agents}, Actions: {num_actions}, Obs size: {obs_size}\n")

        """ LEARNER SETUP """
        if self.agent is None:
            raise ValueError("No trained model found for evaluation.")

        """ EVALUATION LOGIC """
        for episode in range(eval_episodes):
            next_obs, info = env.reset(seed=None)
            total_episodic_return = np.zeros(num_agents)

            for step in range(env.num_steps):
                obs = batchify_obs(next_obs)

                with torch.no_grad():
                    actions, _, _, _ = self.agent.get_action_and_value(obs)

                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                total_episodic_return += batchify(rewards).cpu().numpy()

                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

            if episode == 0:
                # render first episode
                env.render(f"{self.folder}/final_evaluation_{episode + 1}.png")
        
        print(f"Evaluation over {eval_episodes} episodes complete.")
        print(f"Avg Episodic Return: {np.mean(total_episodic_return, axis=0):.2f}")