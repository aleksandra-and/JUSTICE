

import torch
import time
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import supersuit as ss

from thesis_rl.agents.basic_agent import Agent    

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
        self.anneal_lr = True
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.norm_adv = True
        self.clip_vloss = True
        
        self.agent = None
        
        self.total_episodes = args.total_episodes
        self.print_interval = args.backup_interval
        self.num_envs = args.num_envs
        self.num_minibatches = 285 # the same as number of timesteps
        self.update_epochs = 4
        
        timestamp = datetime.now().strftime("%d:%m:%y")
        self.folder = args.save_folder + f"/mappo_{timestamp}"
        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
    
    def train_mappo(self, jenv):
        # Environment variables
        num_agents = len(jenv.possible_agents)
        num_actions = jenv.action_space(jenv.possible_agents[0]).n
        obs_size = jenv.observation_space(jenv.possible_agents[0]).shape[0]
        self.num_steps = jenv.num_years

        print(f"Training MAPPO for {self.total_episodes} with reward: {jenv.reward}")
        print(f"Agents: {num_agents}, Actions: {num_actions}, Obs size: {obs_size}\n")
        run_name = f"MAPPO_{jenv.reward}_{self.total_episodes}eps_{time.time():.0f}"
        writer = SummaryWriter(f"{self.folder}/{run_name}")

        """ LEARNER SETUP """
        self.agent = Agent(num_actions=num_actions, obs_size=obs_size).to(self.device)
        optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        
        """ ENVIRONMENT SETUP """
        env = deepcopy(jenv)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        envs = ss.concat_vec_envs_v1(env, self.num_envs, num_cpus=self.num_envs, base_class="gymnasium")
        envs.single_observation_space = envs.observation_space
        envs.single_action_space = envs.action_space
        envs.is_vector_env = True
        self.batch_size = envs.num_envs * self.num_steps
        self.minibatch_size =  int(self.batch_size // self.num_minibatches)
        
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, envs.num_envs) + envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, envs.num_envs) + envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, envs.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, envs.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, envs.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, envs.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        #print(self.num_envs, envs, env)
        #print(envs.reset())
        next_obs, infos = envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(envs.num_envs).to(self.device)
        num_updates = self.total_episodes

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            """ LOGGING """
            
            # print observations, rewards
            if update % self.print_interval == 0:
                print(f"Update: {update}, Global Step: {global_step}")
                print(f"    Mean Final Return: {rewards[self.num_steps - 1].mean():.5f}")
                print(f"    Mean Episodic Return: {rewards.sum(dim=0).mean():.5f}")
                for obs_idx, observation_names in enumerate(jenv.get_observations_ids()):
                    print(f"    Mean {observation_names}: {obs[-1, :, obs_idx].mean():.5f}")
                print("\n")
                
            writer.add_scalar("charts/mean_episodic_return", rewards.sum(dim=0).mean(), global_step)
            for obs_idx, observation_names in enumerate(jenv.get_observations_ids()):
                writer.add_scalar(f"charts/final_{observation_names}", obs[-1, :, obs_idx].mean(), global_step)
            
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            #print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        writer.close()
        # Final save
        torch.save(self.agent.state_dict(), f"{self.folder}/{run_name}/final_mappo_agent.pt")
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")