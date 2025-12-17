import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools

from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

from justice.model import JUSTICE
from justice.util.enumerations import *
import seaborn as sns

LOCAL_OBSERVATIONS = [  # local observations, not shared with other agents
    "net_economic_output",
    "regional_temperature", # can remove
    "economic_damage",
    "abatement_cost",
    "emissions",
]

GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents

class JusticeEnvironment(ParallelEnv):
    metadata = {
        "name": "justice_environment_v0",
    }

    def __init__(self):
        self.possible_agents = [f"region_{i}" for i in range(1, 5 + 1)]
        self.agents = None
        self.timestep = None
        self.model = JUSTICE(
            scenario=2, # SSP scenarios
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
            climate_ensembles=[1001], # climate uncertainty ensembles
            clustering=True,
            cluster_level=len(self.possible_agents),
            stochastic_run=False,
        )
        self.timestep = None
        self.start_year = 2015
        self.end_year = 2300
        self.emissions_control_rate = None
        self.num_steps = self.end_year - self.start_year
        self.action_change = None
        

    def reset(self, seed=None, options=None):
        # Currently seed not used
        self.seed = seed
        self.possible_agents = [f"region_{i}" for i in range(1, 5 + 1)] 
        self.agents = copy(self.possible_agents)
        self.model.reset() # Reset the model to its initial state
        
        self.timestep = 0
        # self.start_year = 2015
        # self.end_year = 2300
        # self.num_steps = self.end_year - self.start_year
        self.action_change = 1 # regions can change their actions by max 0.1 per step
        self.emissions_control_rate = np.zeros((len(self.agents), self.num_steps))
        
        observations = self.get_observations(self.model.stepwise_evaluate(timestep=self.timestep), None)
        infos = {a: {} for a in self.agents}
        
        return observations, infos
        

    def step(self, actions):
        # Get corresponding actions for all agents
        self.emissions_control_rate[:, self.timestep] = [actions[agent] * 0.25 for agent in self.agents]
        
        # Run the model for the current timestep
        self.model.stepwise_run(emission_control_rate = self.emissions_control_rate[:, self.timestep], timestep=self.timestep, endogenous_savings_rate=True)
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        
        # Get observations, rewards, done, infos
        observations = self.get_observations(data, actions)
        rewards = self.get_rewards(data)
        done = (
            self.timestep >= self.num_steps - 1
        )  # ends when the last year is reached
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {a: {} for a in self.agents}
        
        if self.timestep >= self.num_steps - 1:
            # Remove agents if done
            self.agents = []
        else:
            # Increment timestep after evaluation if the episode is not done
            self.timestep += 1
        
        return observations, rewards, terminated, truncated, infos
        
    def get_observations(self, data, actions):        
        local_obs = np.array(
            [
                data[key][:, self.timestep, :].mean(axis=1)
                for key in LOCAL_OBSERVATIONS
            ],
            dtype=np.float32,
        )
        global_obs = np.array(
            [
                data[key][self.timestep, :].mean(axis=0)
                for key in GLOBAL_OBSERVATIONS
            ],
            dtype=np.float32
        )
        
        observations = {
            agent: np.concatenate((np.array([self.timestep/self.num_steps, i/len(self.agents)], dtype=np.float32), 
                                   local_obs[:, i], global_obs, 
                                   self.emissions_control_rate[:, self.timestep].astype(np.float32)))
            for i, agent in enumerate(self.agents)
        }
        
        return observations
    
    def get_rewards(self, data):
        rewards = {
            agent: 1 / data["regional_temperature"][i, self.timestep] 
            # or stepwise_marl_reward | consumption_per_capita
            for i, agent in enumerate(self.agents)
        }
         
        return rewards

    def render(self, filename='agents_emissions_control_rate.png'):
        if self.emissions_control_rate is None:
            print("No data to render yet.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        years = range(self.start_year, self.start_year + self.emissions_control_rate.shape[1])
        
        # Plot emissions control rate
        for i, agent in enumerate(self.possible_agents):
            ax1.plot(years, self.emissions_control_rate[i, :], label=agent)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Emissions Control Rate')
        ax1.set_title('Emissions Control Rate per Agent')
        ax1.legend()
        ax1.grid(True)
        
        # Plot global temperature
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        global_temp = data['global_temperature'][0:self.num_steps, :]
        
        # If global_temp is 2D (timesteps x ensembles), take mean across ensembles
        if global_temp.ndim > 1:
            global_temp = global_temp.mean(axis=1)
        
        # Create years array matching global_temp length
        temp_years = range(self.start_year, self.start_year + len(global_temp))
        
        ax2.plot(temp_years, global_temp, label='Global Temperature')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Global Temperature (°C)')
        ax2.set_title('Global Temperature Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    2 + # timestep and agent id
                    len(LOCAL_OBSERVATIONS) + len(GLOBAL_OBSERVATIONS) 
                    + len(self.possible_agents),
                ),
                dtype=np.float32,
            )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)  
    
    