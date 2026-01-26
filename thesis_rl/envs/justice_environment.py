# Traninig implementation of PPO from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools

from gymnasium.spaces import Discrete, Box
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
            args = type('obj', (object,), {'num_agents': 5, 'reward': 'consumption_per_capita'})()
        
        self.LOCAL_OBSERVATIONS = [  # local observations, not shared with other agents
            "net_economic_output",
            "regional_temperature", # can remove
            "economic_damage",
            "abatement_cost",
            "emissions",
        ]

        self.GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents
        
        
        self.possible_agents = [f"region_{i}" for i in range(1, args.num_agents + 1)]
        self.render_mode = render_mode
        self.agents = None
        self.timestep = None
        self.model = JUSTICE(
            scenario=2, # SSP scenarios
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
            climate_ensembles=args.ensables, # climate uncertainty ensembles
            clustering=True,
            cluster_level=len(self.possible_agents),
            stochastic_run=False,
        )
        
        self.timestep = None
        self.start_year = 2015
        self.end_year = 2300
        self.agent_emissions_control_rate = None
        self.num_years = self.end_year - self.start_year
        self.action_change = 2  # regions can change their actions by max 0.2 per step
        self.reward = args.reward  # 'global_temperature' 'consumption_per_capita' or 'stepwise_marl_reward'
        

    def reset(self, seed=None, options=None):
        # Currently seed not used
        self.seed = seed 
        self.agents = copy(self.possible_agents)
        self.model.reset() # Reset the model to its initial state
        
        self.timestep = 0
        # self.start_year = 2015
        # self.end_year = 2300
        # self.num_steps = self.end_year - self.start_year
        self.action_change = 3 # regions can change their actions by max 0.3 per step
        self.agent_emissions_control_rate = np.zeros((len(self.possible_agents), self.num_years))
        
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
        self.agent_emissions_control_rate[:, self.timestep] = [actions[agent] * 0.1 for agent in self.agents]
        
        # Convert agent actions to model format
        unmapped_emmissions = np.zeros(57)
        for region_idx, cluster_idx in self.model.country_to_cluster.items():
            unmapped_emmissions[region_idx] = self.agent_emissions_control_rate[cluster_idx, self.timestep]
        
        # Run the model for the current timestep
        self.model.stepwise_run(emission_control_rate = self.agent_emissions_control_rate[:, self.timestep], timestep=self.timestep, endogenous_savings_rate=True)
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        
        # Get observations, rewards, done, infos
        observations = self.get_observations(data, actions)
        rewards = self.get_rewards(data)
        done = (
            self.timestep >= self.num_years - 1
        )  # ends when the last year is reached
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        self.action_mask = {agent: self.get_avail_agent_actions(i) for i, agent in enumerate(self.agents)}
        infos = {
                    a: {
                        'rewards': rewards[a],
                        'mitigated_emissions': self.agent_emissions_control_rate[i, self.timestep],
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
        """Returns the available actions for agent_id"""
        num_actions = self.action_space(agent_idx).n
        if self.timestep == 0:
            return [1] * num_actions
        
        action_mask = [0] * num_actions
        last_action = int(self.agent_emissions_control_rate[agent_idx, self.timestep] * 10)
        
        range_start = max(0, last_action - self.action_change)
        range_end = min(num_actions - 1, last_action + self.action_change + 1)
        
        action_mask[range_start:range_end] = [1] * (range_end - range_start)
        return action_mask
      
    def cluster_observations(self, obs, num_obs):
        """Function clusters regions into """
        clustered_obs = np.zeros((len(self.possible_agents), num_obs))
        if num_obs == 1:
            clustered_obs = np.zeros(len(self.possible_agents))
        
        for region_idx, cluster_idx in self.model.country_to_cluster.items():
            if obs.ndim == 1:
                clustered_obs[cluster_idx] += obs[region_idx]
            else:
                clustered_obs[cluster_idx, :] += obs[:, region_idx]
        
        return clustered_obs
        
    def get_observations(self, data, actions):        
        local_obs = self.cluster_observations(
            np.array(
                [
                    data[key][:, self.timestep, :].mean(axis=1)
                    for key in self.LOCAL_OBSERVATIONS
                ],
                dtype=np.float32,
            ),
            num_obs = len(self.LOCAL_OBSERVATIONS)
        )
        
        global_obs = np.array(
            [
                data[key][self.timestep, :].mean(axis=0)
                for key in self.GLOBAL_OBSERVATIONS
            ],
            dtype=np.float32
        )
       
        observations = {
            agent: np.concatenate((local_obs[i, :], global_obs, 
                                   self.agent_emissions_control_rate[:, self.timestep].astype(np.float32)))
            for i, agent in enumerate(self.agents)
        }
        
        return observations
    
    def get_observations_ids(self):
        return self.LOCAL_OBSERVATIONS + self.GLOBAL_OBSERVATIONS
    
    def get_rewards(self, data):
        
        rewards = {}
        match self.reward:
            case 'regional_temperature':
                observed_reward = data[self.reward][:, self.timestep, :].mean(axis=1)
                observed_reward = self.cluster_observations(observed_reward, num_obs=1)
                
                rewards = {
                    agent: 1.0 / observed_reward[i]
                    for i, agent in enumerate(self.agents)
                }
            case 'global_temperature':
                rewards = {
                    agent: 1.0 / data[self.reward][self.timestep, :].mean()
                    for i, agent in enumerate(self.agents)
                }
            case 'stepwise_marl_reward':
                observed_reward = data[self.reward][:, self.timestep]
                observed_reward = self.cluster_observations(observed_reward, num_obs=1)
            
                rewards = {
                    agent: observed_reward[i]
                    # or stepwise_marl_reward | consumption_per_capita
                    for i, agent in enumerate(self.agents)
                }
            # case 'consumption_per_capita':
            #     observed_reward = data[self.reward][:, self.timestep, :].mean(axis=1)
            #     clustered_consumption = np.zeros(len(self.possible_agents))
                
            #     for region_idx, cluster_idx in self.model.country_to_cluster.items():
            #         clustered_consumption[cluster_idx] += observed_reward[region_idx] * self.model.region_populations[region_idx, self.start_year + self.timestep]
            case _:
                observed_reward = data[self.reward][:, self.timestep, :].mean(axis=1)
                observed_reward = self.cluster_observations(observed_reward, num_obs=1)
                
                rewards = {
                    agent: observed_reward[i]
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
        
        local_obs = self.cluster_observations(
            np.array(
                [
                    data[key][:, self.timestep, :].mean(axis=1)
                    for key in self.LOCAL_OBSERVATIONS
                ],
                dtype=np.float32,
            ),
            num_obs = len(self.LOCAL_OBSERVATIONS)
        )
        
        state = np.concatenate((
            global_obs,
            local_obs.flatten(),
            self.agent_emissions_control_rate[:, self.timestep].astype(np.float32)
        ))
        
        return state
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self.LOCAL_OBSERVATIONS) + len(self.GLOBAL_OBSERVATIONS) +
                    + len(self.possible_agents), # emissions control rates
                ),
                dtype=np.float32,
            )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(11)
    
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self.GLOBAL_OBSERVATIONS)
                    + len(self.LOCAL_OBSERVATIONS) * len(self.possible_agents)
                    + len(self.possible_agents), # emissions control rates
                ),
                dtype=np.float32,
            )
        
    def render(self):
        print("Rendering JUSTICE Environment...")
        
        print(
            f"Timestep: {self.timestep}, Year: {self.start_year + self.timestep}"
        )
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        for i, agent in enumerate(self.agents):
            print(
                f"Agent: {agent}, Emissions Control Rate: {self.agent_emissions_control_rate[i, self.timestep]:.2f}, "
                f"Regional Temperature: {data['regional_temperature'][:, self.timestep, :].mean(axis=1)[self.model.cluster_to_country[i]].mean():.2f}, "
                f"Net Economic Output: {data['net_economic_output'][:, self.timestep, :].mean(axis=1)[self.model.cluster_to_country[i]].mean():.2f}"
            )