# Traninig implementation of PPO from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools

from gymnasium.spaces import Discrete, Box, MultiDiscrete
from pettingzoo import ParallelEnv

from justice.model import JUSTICE
from justice.util.enumerations import *
import seaborn as sns

class JusticeEnvironment(ParallelEnv):
    metadata = {
        "name": "justice_environment_v0",
    }

    def __init__(self, args=None, render_mode=None):
        # For testing purposes
        if args is None:
            args = type('obj', (object,), {'num_agents': 5, 'reward': 'global_temperature'})()
        
        self.LOCAL_OBSERVATIONS = [  # local observations, not shared with other agents
            "net_economic_output",
            "regional_temperature", # can remove
            "economic_damage",
            "abatement_cost",
            "emissions",
        ]

        self.GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents
        
        ### RL config parameters ###
        self.possible_agents = [f"region_{i}" for i in range(1, args.num_agents + 1)]
        self.render_mode = render_mode
        self.agents = None
        self.timestep = None
        self.reward = args.reward  # 'global_temperature' 'consumption_per_capita' or 'stepwise_marl_reward'
        self.state_type = args.state_type
        self.num_actions = args.num_actions # 0.0, 0.05, ..., 1.0 
        self.action_change = args.action_change  # regions can change their actions by max 0.2 per step
        
        # action 1
        self.agent_emissions_control_rate = None 
        # action 2
        self.agent_savings_rate = None 
        
        ### JUSTICE model parameters ###
        self.timestep = None
        self.ensables = args.ensables
        self.fixed_savings_rate = args.fixed_savings_rate
        self.model = JUSTICE(
            scenario=2, # SSP scenarios
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
            climate_ensembles=self.ensables, # climate uncertainty ensebles
            clustering=True,
            cluster_level=len(self.possible_agents),
            stochastic_run=False,
        )
        # Getting population from the model with shape: (57, num_years, num_ensebles)
        self.population = self.model.economy.get_population()
        self.start_year = 2015
        self.end_year = 2300
        self.num_years = self.end_year - self.start_year

    def reset(self, seed=None, options=None):
        # Currently seed not used
        self.seed = seed 
        self.agents = copy(self.possible_agents)
        self.model.reset() # Reset the model to its initial state
        
        # reset timestep and action history
        self.timestep = 0
        self.agent_emissions_control_rate = np.zeros((len(self.possible_agents), self.num_years))
        self.agent_savings_rate = np.zeros((len(self.possible_agents), self.num_years))
        
        observations = self.get_observations(self.model.stepwise_evaluate(timestep=self.timestep), None)
        self.action_mask = {agent: self.get_avail_agent_actions(i) for i, agent in enumerate(self.agents)}
        infos = {
                a: {
                    'rewards': [],
                    'action_mask': self.action_mask[a],
                } 
                for i, a in enumerate(self.agents)
            }
        
        return observations, infos
        

    def step(self, actions):
        # Get corresponding actions for all agents
        # actions[agent] is a 2-element array: [emission_control_action, savings_rate_action]
        self.agent_emissions_control_rate[:, self.timestep] = [
                actions[agent][0] / (self.num_actions - 1) 
            for agent in self.agents
        ]
        self.agent_savings_rate[:, self.timestep] = [
                actions[agent][1] / (self.num_actions - 1) 
            for agent in self.agents
        ]
        
        # Run the model for the current timestep with both emission control and savings rate
        self.model.stepwise_run(
            emission_control_rate=self.agent_emissions_control_rate[:, self.timestep],
            timestep=self.timestep,
            savings_rate=self.agent_savings_rate[:, self.timestep],
            endogenous_savings_rate=False
        )
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        
        # Get observations, rewards, done, infos
        observations = self.get_observations(data, actions)
        rewards = self.get_rewards(data)
        done = (
            self.timestep >= self.num_years - 1
        )  # ends when the last year is reached
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: done for agent in self.agents}
        self.action_mask = {agent: self.get_avail_agent_actions(i) for i, agent in enumerate(self.agents)}
        infos = {
                    a: {
                        'rewards': rewards[a],
                        'mitigated_emissions': self.agent_emissions_control_rate[i, self.timestep],
                        'savings_rate': self.agent_savings_rate[i, self.timestep],
                        'action_mask': self.action_mask[a],
                    } 
                    for i, a in enumerate(self.agents)
                }
        
        if self.timestep >= self.num_years - 1:
            # Remove agents if done
            self.agents = []
        else:
            # Increment timestep after evaluation if the episode is not done
            self.timestep += 1
        
        return observations, rewards, terminated, truncated, infos
    
    def get_avail_agent_actions(self, agent_idx):
        """Returns the available actions for agent_id.
        Returns a flattened array of action masks for MultiDiscrete:
        [emission_control_mask..., savings_rate_mask...]
        """
        action_nvec = self.action_space(agent_idx).nvec  # [num_actions, num_actions]
        
        if self.timestep == 0:
            # All actions available at first timestep
            return np.ones(sum(action_nvec), dtype=np.float32)
        
        # Mask for emission control rate
        emission_mask = np.zeros(action_nvec[0], dtype=np.float32)
        last_emission_action = int(self.agent_emissions_control_rate[agent_idx, self.timestep - 1] * (self.num_actions - 1))
        range_start = max(0, last_emission_action - self.action_change)
        range_end = min(action_nvec[0], last_emission_action + self.action_change + 1)
        emission_mask[range_start:range_end] = 1.0
        
        # Mask for savings rate (same logic)
        savings_mask = np.zeros(action_nvec[1], dtype=np.float32)
        last_savings_action = int(self.agent_savings_rate[agent_idx, self.timestep - 1] * (self.num_actions - 1))
        range_start = max(0, last_savings_action - self.action_change)
        range_end = min(action_nvec[1], last_savings_action + self.action_change + 1)
        savings_mask[range_start:range_end] = 1.0
        
        return np.concatenate([emission_mask, savings_mask])
        
    def get_observations(self, data, actions):        
        local_obs = np.array(
                [
                    data[key][:, self.timestep, :].mean(axis=1)
                    for key in self.LOCAL_OBSERVATIONS
                ],
                dtype=np.float32,
            )
        
        global_obs = np.array(
            [
                data[key][self.timestep, :].mean(axis=0)
                for key in self.GLOBAL_OBSERVATIONS
            ],
            dtype=np.float32
        )
       
        observations = {
            agent: np.concatenate((
                                   local_obs[:, self.model.cluster_to_country[i]].mean(axis=1), # local obs for the agent's cluster
                                   global_obs, 
                                   self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                                   self.agent_savings_rate[:, self.timestep].astype(np.float32)))
            for i, agent in enumerate(self.agents)
        }
        
        return observations
    
    def get_observations_ids(self):
        return self.LOCAL_OBSERVATIONS + self.GLOBAL_OBSERVATIONS
    
    def get_rewards(self, data):
        
        rewards = {}
        match self.reward:
            case 'global_economic_output':
                net_output = data['net_economic_output'][:, self.timestep, :].mean(axis=1).sum()
                
                rewards = {
                    agent: net_output / 1000
                    for i, agent in enumerate(self.agents)
                }
            case 'global_temperature':
                rewards = {
                    agent: 1 / data['global_temperature'][self.timestep, :].mean()
                    for i, agent in enumerate(self.agents)
                }
            case 'consumption_per_capita':
                observed_reward = data[self.reward][:, self.timestep, :].mean(axis=1)
                current_population = self.population[:, self.timestep, :].mean(axis=1)
                
                rewards = {
                    agent: observed_reward[self.model.cluster_to_country[i]].sum() / 
                            current_population[self.model.cluster_to_country[i]].sum()
                    for i, agent in enumerate(self.agents)
                }
            case 'avg_temperature_threshold':
                # ensable memebers below 2 degrees threshold
                below_threshold = np.where(data['global_temperature'][self.timestep, :] <= 2.0, 1.0, 0.0).sum()
                rewards = {
                    agent: below_threshold / len(self.ensables)
                    for i, agent in enumerate(self.agents)
                }
            case 'stepwise_marl_reward':
                observed_reward = data[self.reward][:, self.timestep]
                
                rewards = {
                    agent: observed_reward[self.model.cluster_to_country[i]].sum()
                    # or stepwise_marl_reward | consumption_per_capita
                    for i, agent in enumerate(self.agents)
                }
            case _:
                observed_reward = data[self.reward][:, self.timestep, :].mean(axis=1)
                
                rewards = {
                    agent: observed_reward[self.model.cluster_to_country[i]].sum()
                    # or stepwise_marl_reward | consumption_per_capita
                    for i, agent in enumerate(self.agents)
                }
         
        return rewards

    def get_state(self):
        # Global observations
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        global_obs = np.array(
            [
                data[key][self.timestep, :].mean(axis=0)
                for key in self.GLOBAL_OBSERVATIONS
            ],
            dtype=np.float32
        )
        
        local_obs = np.array(
                [
                    data[key][:, self.timestep, :].mean(axis=1)
                    for key in self.LOCAL_OBSERVATIONS
                ],
                dtype=np.float32,
            )
        # cluster per agent
        local_obs = np.array([local_obs[:, self.model.cluster_to_country[i]].mean(axis=1) for i in range(len(self.possible_agents))])
        
        if self.state_type == 'EP':
            state = [
                np.concatenate((
                    global_obs,
                    local_obs.flatten(),
                    self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                    self.agent_savings_rate[:, self.timestep].astype(np.float32)
                ))
                for _ in range(len(self.possible_agents))
            ]
        if self.state_type == 'FP':
            state = [
                np.concatenate((
                    global_obs,
                    local_obs[i], 
                    self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                    self.agent_savings_rate[:, self.timestep].astype(np.float32)
                ))
                for i in range(len(self.possible_agents))
            ]
        return state
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self.LOCAL_OBSERVATIONS) + len(self.GLOBAL_OBSERVATIONS) +
                    + len(self.possible_agents)  # emissions control rates
                    + len(self.possible_agents),  # savings rates
                ),
                dtype=np.float32,
            )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiDiscrete([self.num_actions, self.num_actions])
    
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        if self.state_type == 'EP':
            return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self.GLOBAL_OBSERVATIONS)
                    + len(self.possible_agents) * len(self.LOCAL_OBSERVATIONS) 
                    + len(self.possible_agents)  # emissions control rates
                    + len(self.possible_agents),  # savings rates
                ),
                dtype=np.float32,
            )
        if self.state_type == 'FP':
            return Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        len(self.GLOBAL_OBSERVATIONS)
                        + len(self.LOCAL_OBSERVATIONS)
                        + len(self.possible_agents)  # emissions control rates
                        + len(self.possible_agents),  # savings rates
                    ),
                    dtype=np.float32,
                )
    
    def plot_observations(self, data):
        print("Plotting observations...")
        years = np.arange(self.start_year, self.end_year)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot regional temperature
        for i, agent in enumerate(self.possible_agents):
            axes[0, 0].plot(
                years,
                data['regional_temperature'][self.model.cluster_to_country[i], :self.num_years, :].mean(axis=0).mean(axis=1),
                label=f"Region {i+1}",
            )
        axes[0, 0].set_xlabel("Year")
        axes[0, 0].set_ylabel("Regional Temperature")
        axes[0, 0].set_title("Regional Temperature over Time")
        axes[0, 0].legend()
        
        # Plot global temperature
        axes[0, 1].plot(
            years,
            data['global_temperature'][:self.num_years, :].mean(axis=1),
            label="Global Temperature",
            color='black'
        )
        axes[0, 1].set_xlabel("Year")
        axes[0, 1].set_ylabel("Global Temperature")
        axes[0, 1].set_title("Global Temperature over Time")
        axes[0, 1].legend()
        
        # Plot net economic output
        for i, agent in enumerate(self.possible_agents):
            axes[1, 0].plot(
                years,
                data['net_economic_output'][self.model.cluster_to_country[i], :self.num_years, :].mean(axis=0).mean(axis=1),
                label=f"Region {i+1}",
            )
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Net Economic Output")
        axes[1, 0].set_title("Net Economic Output over Time")
        axes[1, 0].legend()
        
        # Plot emissions control rates
        for i, agent in enumerate(self.possible_agents):
            axes[1, 1].plot(
                years,
                self.agent_emissions_control_rate[i, :self.num_years],
                label=f"Region {i+1}",
            )
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Emissions Control Rate")
        axes[1, 1].set_title("Emissions Control Rates over Time")
        axes[1, 1].legend()
        
        # Plot savings rates
        for i, agent in enumerate(self.possible_agents):
            axes[2, 0].plot(
                years,
                self.agent_savings_rate[i, :self.num_years],
                label=f"Region {i+1}",
            )
        axes[2, 0].set_xlabel("Year")
        axes[2, 0].set_ylabel("Savings Rate")
        axes[2, 0].set_title("Savings Rates over Time")
        axes[2, 0].legend()
        
        # Hide unused subplot
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("observations_over_time.png")
        plt.close()
        
    def render(self):
        if self.timestep == self.num_years-1:
            data = self.model.stepwise_evaluate(timestep=self.timestep)
            for i, agent in enumerate(self.possible_agents):
                print(
                    f"  Regional Temperature: {data['regional_temperature'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}, "
                    f"  Net Economic Output: {data['net_economic_output'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}"
                )
            
            self.plot_observations(data)