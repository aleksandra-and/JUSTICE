"""
Multi-Objective Justice Environment for MOMARL training.

Extends MOParallelEnv from MOMAland to provide vector rewards for multi-objective
optimization. Compatible with MOMAland wrappers (LinearizeReward, NormalizeReward).
"""

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import functools
import csv
from datetime import datetime
from pathlib import Path

import h5py

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
        "inverse_global_temperature",  # 1 / global_temperature
        "global_economic_output",      # sum of net economic output
        "consumption_per_capita",      # consumption / population
        "gini_consumption",            # Gini coefficient of consumption (lower is better, we negate)
        "temperature_threshold",       # fraction of ensembles below 2C threshold
        "welfare",                     # spatially aggregated welfare
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
                'welfare_type': 'utilitarian',
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
        
        self.welfare_type = getattr(args, 'welfare_type', 'utilitarian')

        social_welfare_function = WelfareFunction.UTILITARIAN
        if self.welfare_type == 'utilitarian':
            social_welfare_function = WelfareFunction.UTILITARIAN
        elif self.welfare_type == 'prioritarian':
            social_welfare_function = WelfareFunction.PRIORITARIAN
        elif self.welfare_type == 'sufficientarian':
            social_welfare_function = WelfareFunction.SUFFICIENTARIAN
        elif self.welfare_type == 'egalitarian':
            social_welfare_function = WelfareFunction.EGALITARIAN
        
        self.model = JUSTICE(
            scenario=2,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=social_welfare_function,
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
        self.start_year = args.start_year
        self.end_year = args.end_year
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

        # Evaluation export configuration.
        self.save_evaluation_data = getattr(args, 'save_evaluation_data', True)
        self.evaluation_output_dir = Path(getattr(args, 'evaluation_output_dir', 'data/temporary/momarl_eval'))
        self.evaluation_scenario_name = 'SSP245'
        self.evaluation_run_tag = getattr(args, 'evaluation_run_tag', self.welfare_type.capitalize())
        self.save_full_hdf5 = True
        self.save_core_npy = True
        self.save_summary_csv = True
        self.core_export_variables = [
                'global_temperature',
                'emissions',
                'constrained_emission_control_rate',
                'net_economic_output',
                'consumption_per_capita',
                'regional_temperature',
            ]
        self._episode_counter = -1
        self._episode_saved = False

    def reset(self, seed=None, options=None):
        self.seed = seed 
        self.agents = copy(self.possible_agents)
        self.model.reset()

        self._episode_counter += 1
        self._episode_saved = False
        
        self.timestep = 0
        self.action_change = 3
        self.agent_emissions_control_rate = np.zeros((len(self.possible_agents), self.num_years+1))
        self.agent_savings_rate = np.zeros((len(self.possible_agents), self.num_years+1))
        
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
                'rewards': np.array(rewards[a]).copy(),
                'temperature': data['global_temperature'][self.timestep, :].mean(),
                'economic_output': data['net_economic_output'][:, self.timestep, :].mean(axis=1).sum(),
                'mitigated_emissions': self.agent_emissions_control_rate[i, self.timestep],
                'savings_rate': self.agent_savings_rate[i, self.timestep],
                'action_mask': self.action_mask[a],
            } 
            for i, a in enumerate(self.agents)
        }
        
        if self.timestep >= self.num_years:
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
        
        # Agent can choose any action in the first step
        if self.timestep == 0:
            if self.fixed_savings_rate:
                return np.ones(self.num_actions, dtype=np.float32)
            return np.ones(self.num_actions * 2, dtype=np.float32)
        
        emission_mask = build_action_mask(
            self.agent_emissions_control_rate[agent_idx, self.timestep],
            self.num_actions
        )
        
        if self.fixed_savings_rate:
            return emission_mask
        
        savings_mask = build_action_mask(
            self.agent_savings_rate[agent_idx, self.timestep],
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
                
                elif obj == 'welfare':
                    r = data["spatially_aggregated_welfare"][self.timestep]
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
                    
                elif obj == 'temperature_threshold':
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

    def _build_constrained_emission_control_rate(self):
        """Expand cluster-level agent actions to full regional tensor for plotting/export."""
        emissions_reference = self.model.data['emissions']
        n_regions, n_timesteps, n_ensembles = emissions_reference.shape

        constrained_emission_control_rate = np.zeros(
            (n_regions, n_timesteps, n_ensembles),
            dtype=np.float32,
        )
        available_timesteps = min(n_timesteps, self.agent_emissions_control_rate.shape[1])

        cluster_mapping = self.model.cluster_to_country
        if isinstance(cluster_mapping, dict):
            cluster_items = sorted(cluster_mapping.items(), key=lambda x: int(x[0]))
        else:
            cluster_items = list(enumerate(cluster_mapping))

        for agent_idx, region_indices in cluster_items:
            if agent_idx >= self.agent_emissions_control_rate.shape[0]:
                continue

            agent_policy = self.agent_emissions_control_rate[agent_idx, :available_timesteps].astype(np.float32)
            region_indices = np.atleast_1d(np.array(region_indices, dtype=np.int64))
            constrained_emission_control_rate[region_indices, :available_timesteps, :] = agent_policy[None, :, None]

        return constrained_emission_control_rate

    def _append_objective_summary(self, datasets, constrained_emission_control_rate):
        """Append episode-level summary metrics for quick model selection."""
        summary_path = self.evaluation_output_dir / 'objective_summary.csv'
        file_exists = summary_path.exists()

        global_temperature = datasets.get('global_temperature')
        years_above_threshold = np.nan
        final_global_temperature = np.nan
        if isinstance(global_temperature, np.ndarray) and global_temperature.ndim == 2:
            mean_global_temperature = np.mean(global_temperature, axis=1)
            years_above_threshold = int(np.sum(mean_global_temperature > 2.0))
            final_timestep = min(self.timestep, global_temperature.shape[0] - 1)
            final_global_temperature = float(np.mean(global_temperature[final_timestep, :]))

        emissions_value = datasets.get('emissions')
        peak_global_emissions = np.nan
        final_global_emissions = np.nan
        if isinstance(emissions_value, np.ndarray) and emissions_value.ndim >= 2:
            mean_global_emissions = np.mean(emissions_value, axis=tuple(range(1, emissions_value.ndim)))
            peak_global_emissions = float(np.max(mean_global_emissions))
            final_timestep = min(self.timestep, emissions_value.shape[1] - 1)
            final_global_emissions = float(np.mean(emissions_value[:, final_timestep, ...]))

        net_economic_output = datasets.get('net_economic_output')
        final_global_net_output = np.nan
        if isinstance(net_economic_output, np.ndarray) and net_economic_output.ndim == 3:
            final_timestep = min(self.timestep, net_economic_output.shape[1] - 1)
            final_global_net_output = float(np.sum(np.mean(net_economic_output[:, final_timestep, :], axis=1)))

        welfare_value = datasets.get('welfare')
        if isinstance(welfare_value, np.ndarray):
            welfare_value = float(np.squeeze(welfare_value))
        elif welfare_value is not None:
            welfare_value = float(welfare_value)

        n_regions = int(constrained_emission_control_rate.shape[0])

        summary_row = {
            'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
            'episode': self._episode_counter,
            'run_tag': self.evaluation_run_tag,
            'scenario': self.evaluation_scenario_name,
            'welfare_type': self.welfare_type,
            'num_agents': len(self.possible_agents),
            'num_objectives': self.num_objectives,
            'objectives': '|'.join(self.rewards_list),
            'welfare': welfare_value,
            'years_above_temperature_threshold': years_above_threshold,
            'final_global_temperature': final_global_temperature,
            'peak_global_emissions': peak_global_emissions,
            'final_global_emissions': final_global_emissions,
            'final_global_net_economic_output': final_global_net_output,
            'disaggregated_regions': n_regions,
            'last_timestep': self.timestep,
            'saved_h5': int(self.save_full_hdf5),
            'saved_npy': int(self.save_core_npy),
            'constrained_emissions_shape': str(constrained_emission_control_rate.shape),
        }

        with open(summary_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary_row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_row)

    def _save_evaluation_outputs(self):
        """Save HDF5 bundle and core .npy arrays compatible with visualization notebook."""
        self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)

        datasets = dict(self.model.evaluate())
        constrained_emission_control_rate = self._build_constrained_emission_control_rate()
        datasets['constrained_emission_control_rate'] = constrained_emission_control_rate

        episode_prefix = f"{self.evaluation_run_tag}_idx{self._episode_counter}"

        if self.save_full_hdf5:
            h5_path = self.evaluation_output_dir / f"{episode_prefix}.h5"
            with h5py.File(h5_path, 'w') as h5_file:
                scenario_group = h5_file.create_group(self.evaluation_scenario_name)
                for key, array in datasets.items():
                    scenario_group.create_dataset(key, data=np.asarray(array))

        if self.save_core_npy:
            for variable_name in self.core_export_variables:
                if variable_name not in datasets:
                    continue
                npy_path = (
                    self.evaluation_output_dir
                    / f"{episode_prefix}_{self.evaluation_scenario_name}_{variable_name}.npy"
                )
                np.save(npy_path, np.asarray(datasets[variable_name]))

        if self.save_summary_csv:
            self._append_objective_summary(datasets, constrained_emission_control_rate)

        return datasets
    
    def render(self):
        if self.timestep == self.num_years - 1:
            data = self.model.stepwise_evaluate(timestep=self.timestep)
            # for i, agent in enumerate(self.possible_agents):
            #     print(
            #         f"  Regional Temperature: {data['regional_temperature'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}, "
            #         f"  Net Economic Output: {data['net_economic_output'][self.model.cluster_to_country[i], self.timestep, :].mean(axis=1).mean():.2f}"
            #     )

            if self.save_evaluation_data and not self._episode_saved:
                print(f"Saving evaluation outputs in file {self.evaluation_output_dir} with tag {self.evaluation_run_tag}...")
                self._save_evaluation_outputs()
                self._episode_saved = True
