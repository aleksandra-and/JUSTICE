
# from platypus import NSGAII, Problem, Real
# from solvers.moea.borgMOEA import BorgMOEA
# import time
# def schaffer(x):
#     return [x[0]**2, (x[0]-2)**2]

# problem = Problem(1, 2)
# problem.types[:] = Real(-10, 10)
# problem.function = schaffer

# algorithm = NSGAII(problem)
# # algorithm = BorgMOEA(problem, epsilons=[0.1, 0.1], population_size=50)

# # Measure the time taken to run the algorithm
# start_time = time.time()
# algorithm.run(10000)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# # Display the results.
# for solution in algorithm.result:
#     print(solution.variables, end=" -> ")
#     # print(solution.objectives)


# run_schaffer_with_borg.py

from platypus import Problem, Real
from borg_platypus_adapter import BorgMOEA
import time

def schaffer(x):
    return [x[0]**2, (x[0]-2)**2]

problem = Problem(1, 2)
problem.types[:] = Real(-10, 10)
problem.function = schaffer

# Point this to your compiled library if it isn't in the current directory
# e.g., borg_library_path="./libborg.dylib" (macOS) or "./libborg.so" (Linux)
algorithm = BorgMOEA(
    problem,
    epsilons=[0.1, 0.1],
    population_size=50,
    borg_library_path="./libborg.dylib",  # or "./libborg.so" on Linux; remove if already found
    seed=1,
    solve_settings={
        # Optional: override any Borg parameters here
        # "maximumPopulationSize": 10000,
        # "selectionRatio": 0.02,
        # "restartMode": 0,
        # "frequency": 10000,
        # "runtimefile": "runtime.csv",
        # "runtimeformat": "optimizedv",
    }
)

# Measure the time taken to run the algorithm
start_time = time.time()
algorithm.run(10000)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

for solution in algorithm.result:
    print(solution.objectives)

















