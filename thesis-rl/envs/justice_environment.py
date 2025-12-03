import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools

from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

from justice.model import JUSTICE
from justice.util.enumerations import *

LOCAL_OBSERVATIONS = [  # local observations, not shared with other agents
    "net_economic_output",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]

GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents

class JusticeEnvironment(ParallelEnv):
    metadata = {
        "name": "justice_environment_v0",
    }

    def __init__(self):
        self.possible_agents = None
        self.agents = None
        self.timestep = None
        self.model = None
        self.timestep = None
        self.start_year = None
        self.end_year = None
        self.num_years_per_step = None
        self.emissions_control_rate = None
        self.num_steps = None
        self.action_change = None
        

    def reset(self, seed=None, options=None):
        # Currently seed not used
        self.seed = seed
        self.possible_agents = [f"region_{i}" for i in range(1, 5 + 1)] 
        self.agents = copy(self.possible_agents)
        self.model = JUSTICE(
            scenario=2,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
            climate_ensembles=None,
            clustering=True,
            cluster_level=len(self.agents),
        )
        
        self.timestep = 0
        self.start_year = 2015
        self.end_year = 2100
        self.num_years_per_step = 1
        self.num_steps = (self.end_year - self.start_year) // self.num_years_per_step + 1
        self.action_change = 1 # regions can change their actions by max 0.1 per step
        self.emissions_control_rate = np.zeros((len(self.agents), self.num_steps))
        
        observations = self._get_observations(self.model.stepwise_evaluate(timestep=self.timestep), None)
        infos = {a: {} for a in self.agents}
        
        return observations, infos
        

    def step(self, actions):
        # Get corresponding actions for all agents
        self.emissions_control_rate[:, self.timestep] = [a / 10 for a in actions.values()]
        
        # Run the model for the current timestep
        self.model.stepwise_run(emission_control_rate = self.emissions_control_rate[:, self.timestep], timestep=self.timestep, endogenous_savings_rate=True)
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        
        # Get observations, rewards, done, infos
        observations = self._get_observations(data, actions)
        rewards = self._get_rewards(data)
        done = (
            self.timestep >= self.num_steps - 1
        )  # ends when the last year is reached
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {a: {} for a in self.agents}
        
        # Remove agents if done
        if self.timestep >= self.num_steps - 1:
            self.agents = []
        
        # Increment timestep after evaluation 
        self.timestep += 1 
        
        return observations, rewards, terminated, truncated, infos

    def _get_observations(self, data, actions):
        observations = {
            agent: {"observation": [], "action_mask": []}
            for agent in self.agents
        }
        
        for i, agent in enumerate(self.agents):
            num_actions = self.action_space(agent).n
            
            # Create action mask based on current action
            action_mask = np.zeros(num_actions, dtype=np.int8)
            if actions is not None:
                action_mask[max(0, actions[agent] - self.action_change):
                    min(num_actions, actions[agent] + self.action_change)] = 1
            observations[agent]["action_mask"] = action_mask.tolist()
            
            for local_obs in LOCAL_OBSERVATIONS:
                observations[agent]["observation"].append(data[local_obs][i, self.timestep])
                
            for global_obs in GLOBAL_OBSERVATIONS:
                observations[agent]["observation"].append(data[global_obs][self.timestep])
                
        return observations
    
    def _get_rewards(self, data):
        rewards = {}
        
        for i, agent in enumerate(self.agents):
            rewards[agent] = data["stepwise_marl_reward"][i, self.timestep]
            
        return rewards

    def render(self):
        if self.emissions_control_rate is None:
            print("No data to render yet.")
            return
        
        plt.figure(figsize=(10, 6))
        years = range(self.start_year, self.start_year + self.emissions_control_rate.shape[1])
        
        for i, agent in enumerate(self.agents):
            plt.plot(years, self.emissions_control_rate[i, :], label=agent, marker='o')
        
        plt.xlabel('Year')
        plt.ylabel('Emissions Control Rate')
        plt.title('Emissions Control Rate per Agent')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'timestep_{self.timestep}.png')
        plt.close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(LOCAL_OBSERVATIONS) + len(GLOBAL_OBSERVATIONS),
                    1
                ),
                dtype=np.float32,
            )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(10)  
    
    