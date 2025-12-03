from pettingzoo.test import parallel_api_test
from justice_environment import JusticeEnvironment

if __name__ == "__main__":
    env = JusticeEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)