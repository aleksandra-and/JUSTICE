"""
PPO training for Justice environment. 
Copied from PettingZoo PPO example.
https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/
"""

import numpy as np
import torch
import torch.optim as optim
import time

from envs.justice_environment import JusticeEnvironment
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


if __name__ == "__main__":
    # Hyperparameters
    device = torch.device("cpu")
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    minibatch_size = 128
    update_epochs = 4
    total_episodes = 1000
    print_interval = 100
    folder = "exp_results/recent_run"

    # Environment setup
    env = JusticeEnvironment()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    print(f"Agents: {num_agents}, Actions: {num_actions}, Obs size: {obs_size}\n")

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions, obs_size=obs_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
    max_cycles = env.num_steps

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, obs_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    start_time = time.time()
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

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
        
        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
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
            for start in range(0, len(b_obs), minibatch_size):
                # select the indices we want to train on
                end = start + minibatch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if episode % print_interval == 0:
            print(f"Training episode {episode}/{total_episodes}")
            print(f"    Episodic Return: {np.mean(total_episodic_return):.2f}")
            print(f"    Avg episode time: {(time.time() - start_time)/(episode+1):.2f} seconds")
            print("")
            print(f"    Value Loss: {v_loss.item()}")
            print(f"    Policy Loss: {pg_loss.item()}")
            print(f"    Old Approx KL: {old_approx_kl.item()}")
            print(f"    Approx KL: {approx_kl.item()}")
            print(f"    Clip Fraction: {np.mean(clip_fracs)}")
            print(f"    Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------")
        # Save checkpoint
        if episode % print_interval == 0 and episode > 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{folder}/checkpoint_ep{episode}.pt")

    # Final save
    torch.save(agent.state_dict(), f"{folder}/justice_ppo_final.pt")
    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
    
    """ FINAL EVALUATION """
    torch.load(f"{folder}/justice_ppo_final.pt", weights_only=True)
    with torch.no_grad():
    
        obs, infos = env.reset(seed=None)
        obs = batchify_obs(obs)
        terms = [False]
        truncs = [False]
        episode_return = 0
        step_count = 0
        
        while not any(terms) and not any(truncs):
            actions, _, _, _ = agent.get_action_and_value(obs)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            obs = batchify_obs(obs)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]
            
            episode_return += sum(rewards.values())
            step_count += 1
        
        print(f"Eval: Return = {episode_return}, Steps = {step_count}")
        
        env.render(filename=f"{folder}/final_evaluation.png")
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(total_episodic_return)
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Learning curve saved")