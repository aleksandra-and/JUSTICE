from envs.justice_environment import JusticeEnvironment
from justice.model import JUSTICE
from justice.util.enumerations import *
import numpy as np

if __name__ == "__main__":
    import time

    env = JusticeEnvironment()
    observations, infos = env.reset()

    num_episodes = 50
    start_time = time.time()

    for episode in range(num_episodes):
        observations, infos = env.reset()
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, done, truncated, infos = env.step(actions)
            
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total episodes: {num_episodes}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Avg Episode Time: {elapsed_time/num_episodes:.2f} seconds")
    
    from thesis_rl.train import TrainArgs, EnvArgs
    from thesis_rl.algorithms.mappo import MAPPO
    start_time = time.time()
    train_args = TrainArgs()
    train_args.total_episodes = num_episodes
    train_args.backup_interval = num_episodes + 1  # No backups
    env_args = EnvArgs(reward=train_args.reward, num_agents=train_args.num_agents)
    env = JusticeEnvironment(env_args)
    trainer = MAPPO(train_args)

    trainer.train_mappo(env)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total episodes: {num_episodes}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Avg Episode Time: {elapsed_time/num_episodes:.2f} seconds")
    
    # model = JUSTICE(
    #     scenario=2, # SSP scenarios
    #     economy_type=Economy.NEOCLASSICAL,
    #     damage_function_type=DamageFunction.KALKUHL,
    #     abatement_type=Abatement.ENERDATA,
    #     social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
    #     climate_ensembles=None, # climate uncertainty ensembles
    #     clustering=True,
    #     cluster_level=5,
    #     stochastic_run=False,
    # )
  
    # start_time = time.time()
    # for episode in range(num_episodes):
    #     model.reset()
    #     emission_rate = np.zeros((5, 286))
    #     for timestep in range(286):
    #         emission_rate[:, timestep] = np.random.rand(5)
    #         model.stepwise_run(emission_control_rate=emission_rate[:, timestep], timestep=timestep, endogenous_savings_rate=True)
    #         data = model.stepwise_evaluate(timestep=timestep)
            
    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # print(f"Total episodes: {num_episodes}")
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # model = JUSTICE(
    #     scenario=2, # SSP scenarios
    #     economy_type=Economy.NEOCLASSICAL,
    #     damage_function_type=DamageFunction.KALKUHL,
    #     abatement_type=Abatement.ENERDATA,
    #     social_welfare_function=WelfareFunction.UTILITARIAN,  # WelfareFunction.UTILITARIAN,
    #     climate_ensembles=None, # climate uncertainty ensembles
    #     clustering=True,
    #     cluster_level=5,
    #     stochastic_run=False,
    # )
    # start_time = time.time()
    # for episode in range(num_episodes):
    #     model.reset()
    #     emission_rate = np.array([np.random.rand(286) for _ in range(57)])
    #     model.run(emission_control_rate=emission_rate, endogenous_savings_rate=True)
        
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    
    # print(f"Total episodes: {num_episodes}")
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        