"""
Multi-Objective Justice Environment for MOMARL training.

Extends MOParallelEnv from MOMAland to provide vector rewards for multi-objective
optimization. Compatible with MOMAland wrappers (LinearizeReward, NormalizeReward).
"""

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools

from gymnasium.spaces import Discrete, Box, MultiDiscrete
from momaland.utils.env import MOParallelEnv
from wandb import agent

from justice.model import JUSTICE
from justice.util.enumerations import *


class JusticeEnvironmentMOMA(MOParallelEnv):
    """
    Multi-objective version of (MA) JusticeEnvironment.
    
    Returns vector rewards instead of scalar rewards, compatible with MOMAland.
    Supports configurable objectives via env_args['rewards'] list.
    """
    
    metadata = {
        "name": "justice_environment_mo_v0",
    }

    # Available objectives
    SUPPORTED_OBJECTIVES = [
        "inverse_global_temperature",  # 1 / global_temperature (higher is better)
        "global_economic_output",      # sum of net economic output (higher is better)
        "consumption_per_capita",      # consumption / population (higher is better)
        "gini_consumption",            # Gini coefficient of consumption (lower is better, we negate)
        "avg_temperature_threshold",   # fraction of ensembles below 2C threshold
    ]

    def __init__(self, args=None, render_mode=None):
        # For testing purposes
        if args is None:
            args = type('obj', (object,), {
                'num_agents': 5, 
                'rewards': ['inverse_global_temperature', 'global_economic_output'],
                'ensables': [500],
                'state_type': 'EP',
                'num_actions': 21,
                'action_change': 3,
                'fixed_savings_rate': True,
            })()
        
        self.LOCAL_OBSERVATIONS = [
            "net_economic_output",
            "regional_temperature",
            "economic_damage",
            "abatement_cost",
            "emissions",
        ]

        self.GLOBAL_OBSERVATIONS = ["global_temperature"]
        
        self.possible_agents = [f"region_{i}" for i in range(1, args.num_agents + 1)]
        self.render_mode = render_mode
        self.agents = None
        self.timestep = None
        self.ensables = args.ensables
        
        self.model = JUSTICE(
            scenario=2,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,
            climate_ensembles=self.ensables,
            clustering=True,
            cluster_level=len(self.possible_agents),
            stochastic_run=False,
        )
        
        self.population = self.model.economy.get_population()
        self.state_type = args.state_type
        self.num_actions = args.num_actions
        self.fixed_savings_rate = args.fixed_savings_rate
        
        self.timestep = None
        self.start_year = 2015
        self.end_year = 2300
        self.agent_emissions_control_rate = None
        self.agent_savings_rate = None
        self.num_years = self.end_year - self.start_year
        self.action_change = args.action_change
        
        # Multi-objective configuration
        self.rewards_list = args.rewards if hasattr(args, 'rewards') else ['inverse_global_temperature', 'global_economic_output']
        self.num_objectives = len(self.rewards_list)
        
        # Validate objectives
        for obj in self.rewards_list:
            if obj not in self.SUPPORTED_OBJECTIVES:
                raise ValueError(f"Unsupported objective: {obj}. Supported: {self.SUPPORTED_OBJECTIVES}")

    def reset(self, seed=None, options=None):
        self.seed = seed 
        self.agents = copy(self.possible_agents)
        self.model.reset()
        
        self.timestep = 0
        self.action_change = 3
        self.agent_emissions_control_rate = np.zeros((len(self.possible_agents), self.num_years))
        self.agent_savings_rate = np.zeros((len(self.possible_agents), self.num_years))
        
        observations = self.get_observations(self.model.stepwise_evaluate(timestep=self.timestep), None)
        self.action_mask = {agent: self.get_avail_agent_actions(i) for i, agent in enumerate(self.agents)}
        infos = {
            a: {
                'rewards': np.zeros(self.num_objectives),
                'action_mask': self.action_mask[a],
            } 
            for i, a in enumerate(self.agents)
        }
        
        return observations, infos

    def step(self, actions):
        # Get corresponding actions for all agents
        if self.fixed_savings_rate:
            # Single action: [emission control]
            self.agent_emissions_control_rate[:, self.timestep] = [
                    actions[agent][0] / (self.num_actions - 1)
                for agent in self.agents
            ]
            # Use endogenous savings rate from model
            self.model.stepwise_run(
                emission_control_rate=self.agent_emissions_control_rate[:, self.timestep],
                timestep=self.timestep,
                endogenous_savings_rate=True
            )
            # Store the model's fixed savings rate for observations (clustered average)
            for i in range(len(self.possible_agents)):
                self.agent_savings_rate[i, self.timestep] = self.model.savings_rate[
                    self.model.cluster_to_country[i], self.timestep
                ].mean()
        else:
            # Two actions: [emission_control_action, savings_rate_action]
            self.agent_emissions_control_rate[:, self.timestep] = [
                actions[agent][0] / (self.num_actions - 1)
                for agent in self.agents
            ]
            self.agent_savings_rate[:, self.timestep] = [
                actions[agent][1] / (self.num_actions - 1)
                for agent in self.agents
            ]
            self.model.stepwise_run(
                emission_control_rate=self.agent_emissions_control_rate[:, self.timestep],
                timestep=self.timestep,
                savings_rate=self.agent_savings_rate[:, self.timestep],
                endogenous_savings_rate=False
            )
        
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        
        # Get observations, vector rewards, done, infos
        observations = self.get_observations(data, actions)
        rewards = self.get_rewards(data)  # Now returns vector rewards
        
        done = (self.timestep >= self.num_years - 1)
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: done for agent in self.agents}
        
        self.action_mask = {agent: self.get_avail_agent_actions(i) for i, agent in enumerate(self.agents)}
        infos = {
            a: {
                'rewards': np.array(rewards[a]).copy(),  # Copy to avoid modification by wrappers
                'mitigated_emissions': self.agent_emissions_control_rate[i, self.timestep],
                'savings_rate': self.agent_savings_rate[i, self.timestep],
                'action_mask': self.action_mask[a],
            } 
            for i, a in enumerate(self.agents)
        }
        
        if self.timestep >= self.num_years - 1:
            self.agents = []
        else:
            self.timestep += 1
        
        return observations, rewards, terminated, truncated, infos
    
    def get_avail_agent_actions(self, agent_idx):
        """Returns the available actions for agent_id.
        For fixed_savings_rate=True: returns mask for emission control only.
        For fixed_savings_rate=False: returns flattened mask [emission_mask..., savings_mask...].
        """
        def build_action_mask(last_action, num_actions):
            """Build action mask allowing actions within action_change of last action."""
            if self.timestep == 0:
                return np.ones(num_actions, dtype=np.float32)
            mask = np.zeros(num_actions, dtype=np.float32)
            last_idx = int(last_action * (self.num_actions - 1))
            start = max(0, last_idx - self.action_change)
            end = min(num_actions, last_idx + self.action_change + 1)
            mask[start:end] = 1.0
            return mask
        
        emission_mask = build_action_mask(
            self.agent_emissions_control_rate[agent_idx, max(0, self.timestep - 1)],
            self.num_actions
        )
        
        if self.fixed_savings_rate:
            return emission_mask
        
        savings_mask = build_action_mask(
            self.agent_savings_rate[agent_idx, max(0, self.timestep - 1)],
            self.num_actions
        )
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
                local_obs[:, self.model.cluster_to_country[i]].mean(axis=1),
                global_obs, 
                self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                self.agent_savings_rate[:, self.timestep].astype(np.float32)
            ))
            for i, agent in enumerate(self.agents)
        }
        
        return observations
    
    def get_rewards(self, data):
        """
        Returns vector rewards for each agent.
        
        Each agent receives a numpy array of shape (num_objectives,) containing
        the reward for each objective.
        """
        rewards = {}
        
        for i, agent in enumerate(self.agents):
            agent_rewards = []
            
            for obj in self.rewards_list:
                if obj == 'inverse_global_temperature':
                    # Higher is better (minimize temperature)
                    r = 1.0 / data['global_temperature'][self.timestep, :].mean()
                    agent_rewards.append(r)
                    
                elif obj == 'global_economic_output':
                    # Sum of all regional net economic output, scaled
                    r = data['net_economic_output'][:, self.timestep, :].mean(axis=1).sum() / 1000.0
                    agent_rewards.append(r)
                    
                elif obj == 'consumption_per_capita':
                    # Regional consumption per capita for this agent's cluster
                    consumption = data['consumption_per_capita'][:, self.timestep, :].mean(axis=1)
                    current_population = self.population[:, self.timestep, :].mean(axis=1)
                    r = (consumption[self.model.cluster_to_country[i]].sum() / 
                         current_population[self.model.cluster_to_country[i]].sum())
                    agent_rewards.append(r)
                    
                elif obj == 'gini_consumption':
                    # Gini coefficient of consumption (negated so higher is better = more equal)
                    consumption = data['consumption_per_capita'][:, self.timestep, :].mean(axis=1)
                    gini = self._compute_gini(consumption)
                    r = -gini  # Negate so that lower inequality = higher reward
                    agent_rewards.append(r)
                    
                elif obj == 'avg_temperature_threshold':
                    # Fraction of ensemble members below 2C threshold
                    below_threshold = np.where(data['global_temperature'][self.timestep, :] <= 2.0, 1.0, 0.0).sum()
                    r = below_threshold / len(self.ensables)
                    agent_rewards.append(r)
            
            rewards[agent] = np.array(agent_rewards, dtype=np.float32)
        
        return rewards
    
    def _compute_gini(self, values):
        """Compute Gini coefficient for inequality measurement."""
        values = np.sort(values)
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

    def get_state(self):
        """Returns global state for centralized training."""
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
        local_obs = np.array([
            local_obs[:, self.model.cluster_to_country[i]].mean(axis=1) 
            for i in range(len(self.possible_agents))
        ])
        
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
        elif self.state_type == 'FP':
            state = [
                np.concatenate((
                    global_obs,
                    local_obs[i],
                    self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                    self.agent_savings_rate[:, self.timestep].astype(np.float32)
                ))
                for i in range(len(self.possible_agents))
            ]
        else:
            state = [
                np.concatenate((
                    global_obs,
                    local_obs.flatten(),
                    self.agent_emissions_control_rate[:, self.timestep].astype(np.float32),
                    self.agent_savings_rate[:, self.timestep].astype(np.float32)
                ))
                for _ in range(len(self.possible_agents))
            ]
        return state
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                len(self.LOCAL_OBSERVATIONS) + len(self.GLOBAL_OBSERVATIONS) +
                len(self.possible_agents)  # emissions control rates
                + len(self.possible_agents),  # savings rates
            ),
            dtype=np.float32,
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.fixed_savings_rate:
            # Single action: emission control only
            return Discrete(self.num_actions)
        else:
            # Two actions: emission control and savings rate
            return MultiDiscrete([self.num_actions, self.num_actions])
    
    @functools.lru_cache(maxsize=None)
    def reward_space(self, agent):
        """
        Returns the reward space for multi-objective rewards.
        Required by MOMAland's MOParallelEnv interface.
        """
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_objectives,),
            dtype=np.float32,
        )
    
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        if self.state_type == 'EP':
            return Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self.GLOBAL_OBSERVATIONS)
                    + len(self.LOCAL_OBSERVATIONS) * len(self.possible_agents)
                    + len(self.possible_agents)  # emissions control rates
                    + len(self.possible_agents),  # savings rates
                ),
                dtype=np.float32,
            )
        else:  # FP or default
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
    
    def render(self):
        if self.timestep == self.num_years - 1:
            data = self.model.stepwise_evaluate(timestep=self.timestep)
            for i, agent in enumerate(self.possible_agents):
                print(
                    f"  Regional Temperature: {data['regional_temperature'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}, "
                    f"  Net Economic Output: {data['net_economic_output'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}"
                )
