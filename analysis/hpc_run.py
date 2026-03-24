from run_optimization import run_optimization_momadps, run_single_agent_momadps
import sys
import random
import numpy as np
from justice.util.enumerations import Optimizer, Evaluator

CONFIG_PATH = "analysis/momadps_config.json"
NASH_PROFILES_PATH = "pareto_nash_profiles.csv"
POLICY_BANK_PATH = "COMBINED_MOMA_epsilon_nondominated_set.csv"

if __name__ == "__main__":
    nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # unused, keep for compatibility
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    scenario_index = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    policy_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    variable_macro_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    population_size = int(sys.argv[7]) if len(sys.argv) > 7 else 100

    random.seed(seed)
    np.random.seed(seed)

    run_single_agent_momadps(
        config_path=CONFIG_PATH,
        nash_profiles_path=NASH_PROFILES_PATH,
        policy_bank_path=POLICY_BANK_PATH,
        policy_index=policy_index,
        variable_macro_index=variable_macro_index,
        nfe=nfe,
        seed=seed,
        datapath="./data",
        optimizer=Optimizer.MMBorgMOEA,
        population_size=population_size,
        reference_ssp_rcp_scenario_index=scenario_index,
        evaluator=Evaluator.SequentialEvaluator,
        epsilons=[1e-3, 1e-3],
    )


################

# from run_optimization import run_optimization_adaptive
# import sys
# import random
# import numpy as np
# from justice.util.enumerations import Optimizer, Evaluator

# config_path = "analysis/normative_uncertainty_optimization.json"  # This loads the config used in the Paper

# if __name__ == "__main__":
#     nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # default value 5
#     swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # default value 0
#     seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5000  # default value 5000
#     scenario_index = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # default value 2

#     # Setting the seed
#     random.seed(seed)
#     np.random.seed(seed)
#     run_optimization_adaptive(
#         config_path=config_path,
#         nfe=nfe,
#         swf=swf,
#         seed=seed,
#         datapath="./data",
#         optimizer=Optimizer.MMBorgMOEA,  # Optimizer.BorgMOEA, Optimizer.EpsNSGAII
#         population_size=100,
#         reference_ssp_rcp_scenario_index=scenario_index,
#         evaluator=Evaluator.SequentialEvaluator,
#     )


######################

# from analyzer import run_optimization_momadps
# import sys
# import random
# import numpy as np
# from justice.util.enumerations import Optimizer, Evaluator

# config_path = "analysis/momadps_config.json"  # This loads the config used in the Paper

# if __name__ == "__main__":
#     nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # default value 5
#     swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # default value 0
#     seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5000  # default value 5000
#     scenario_index = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # default value 2

#     # Setting the seed
#     random.seed(seed)
#     np.random.seed(seed)
#     run_optimization_momadps(
#         config_path=config_path,
#         nfe=nfe,
#         seed=seed,
#         datapath="./data",
#         optimizer=Optimizer.MMBorgMOEA,  # Optimizer.BorgMOEA, Optimizer.EpsNSGAII
#         population_size=100,
#         reference_ssp_rcp_scenario_index=scenario_index,
#         evaluator=Evaluator.SequentialEvaluator,
#     )

#######################################################

# from analyzer import run_optimization_momadps, run_single_agent_momadps
# import sys
# import random
# import numpy as np
# from justice.util.enumerations import Optimizer, Evaluator

# CONFIG_PATH = "analysis/momadps_config.json"
# REFERENCE_SET_PATH = "data/temporary/MOMA_DATA/200k/MOMA_reference_set.csv"

# if __name__ == "__main__":
#     nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5
#     swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # unused, keep for compatibility
#     seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
#     scenario_index = int(sys.argv[4]) if len(sys.argv) > 4 else 2
#     policy_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
#     variable_macro_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
#     population_size = int(sys.argv[7]) if len(sys.argv) > 7 else 100

#     random.seed(seed)
#     np.random.seed(seed)

#     run_single_agent_momadps(
#         config_path=CONFIG_PATH,
#         reference_set_path=REFERENCE_SET_PATH,
#         policy_index=policy_index,
#         variable_macro_index=variable_macro_index,
#         nfe=nfe,
#         seed=seed,
#         datapath="./data",
#         optimizer=Optimizer.MMBorgMOEA,
#         population_size=population_size,
#         reference_ssp_rcp_scenario_index=scenario_index,
#         evaluator=Evaluator.SequentialEvaluator,
#         epsilons=[1e-3, 1e-3],
#     )
#######################################################
