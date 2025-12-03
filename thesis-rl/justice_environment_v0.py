from envs.justice_environment import JusticeEnvironment

env = JusticeEnvironment()
observations, infos = env.reset(seed=42)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if env.timestep % 10 == 0:
        env.render()
env.close()