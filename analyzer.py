import functools
import datetime
import numpy as np
import os
import random
from justice.util.enumerations import *
import json

# from solvers.moea.borgMOEA import BorgMOEA
from ema_workbench.em_framework.optimization import EpsNSGAII

# Suppress numpy version warnings
import warnings

warnings.filterwarnings("ignore")

# EMA
from ema_workbench import (
    Model,
    RealParameter,
    ArrayOutcome,
    ScalarOutcome,
    CategoricalParameter,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    MPIEvaluator,
    Constant,
    Scenario,
)
from ema_workbench.util.utilities import save_results, load_results
from ema_workbench.em_framework.optimization import (
    ArchiveLogger,
    EpsilonProgress,
    # HyperVolume,
)

# JUSTICE
# Set this path to the src folder
# export PYTHONPATH=$PYTHONPATH:/Users/palokbiswas/Desktop/pollockdevis_git/JUSTICE/src
# from src.util.enumerations import Scenario
from justice.util.EMA_model_wrapper import (
    model_wrapper,
    model_wrapper_emodps,
    model_wrapper_static_optimization,
)
from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader

from justice.util.enumerations import WelfareFunction, get_welfare_function_name
from config.default_parameters import SocialWelfareDefaults

SMALL_NUMBER = 1e-9  # Used to avoid division by zero in RBF calculations

# EMA Platypus Adapter
from borg_platypus_adapter import BorgMOEA, set_ema_context

from platypus import Solution
from borg_platypus_adapter import _ArchiveView, _AlgorithmStub


# Minimal MPI subclasses that call solveMPI instead of solve()


class MSBorgMOEA(BorgMOEA):
    """Master–Slave Borg: requires MPI and >= 2 ranks."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        # use master–slave library by default
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgms.so",
            solve_settings={},  # no runtime options by default
            **kwargs,
        )

    def run(self, max_evaluations: int):
        from borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)
        if self.seed is not None:
            Configuration.seed(self.seed)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)

        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        # Run MS (islands=1)
        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(islands=1, maxEvaluations=int(max_evaluations))
        finally:
            Configuration.stopMPI()

        # Collect results (only one rank returns a non-None result)
        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                sol = Solution(self.problem)
                sol.variables = list(s_borg.getVariables())
                sol.objectives = list(s_borg.getObjectives())
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = int(max_evaluations)
        else:
            self.nfe = 0

        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)


class MMBorgMOEA(BorgMOEA):
    """Multi‑Master Borg: requires MPI and P = islands*(K+1)+1 ranks."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        import os

        islands = int(os.environ.get("BORG_ISLANDS", "2"))  # override via env var
        # store islands in solve_settings if you want to read later (not required here)
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgmm.so",
            solve_settings={"islands": islands},
            **kwargs,
        )
        self._islands = islands

    def run(self, max_evaluations: int):
        from borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)
        if self.seed is not None:
            Configuration.seed(self.seed)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)

        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        # Run MM with islands = self._islands
        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=self._islands, maxEvaluations=int(max_evaluations)
            )
        finally:
            Configuration.stopMPI()

        # Collect results (only one rank returns a non-None result)
        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                sol = Solution(self.problem)
                sol.variables = list(s_borg.getVariables())
                sol.objectives = list(s_borg.getObjectives())
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = int(max_evaluations)
        else:
            self.nfe = 0

        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)


def run_optimization_adaptive(
    config_path,
    nfe=None,
    population_size=100,  # Default population size. For local machine, use smaller values like 5 or less
    swf=0,
    seed=None,
    datapath="./data",
    filename=None,
    folder=None,
    economy_type=Economy.NEOCLASSICAL,
    damage_function_type=DamageFunction.KALKUHL,
    abatement_type=Abatement.ENERDATA,
    optimizer=Optimizer.EpsNSGAII,
    evaluator=Evaluator.SequentialEvaluator,
):

    # Load configuration from file
    with open(config_path, "r") as file:
        config = json.load(file)

    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]
    n_rbfs = config["n_rbfs"]
    n_inputs = config["n_inputs"]
    epsilons = config["epsilons"]
    temperature_year_of_interest = config["temperature_year_of_interest"]
    reference_ssp_rcp_scenario_index = config["reference_ssp_rcp_scenario_index"]

    stochastic_run = config["stochastic_run"]

    # Check if climate_ensemble_members is in the config, if not set it to None
    if "climate_ensemble_members" in config:
        climate_ensemble_members = config["climate_ensemble_members"]
    else:
        climate_ensemble_members = None

    social_welfare_function = WelfareFunction.from_index(swf)
    social_welfare_function_type = social_welfare_function.value[
        0
    ]  # Gets the first value of the tuple with index 0

    model = Model("JUSTICE", function=model_wrapper_emodps)

    # Instantiate classes and compute derived parameters
    data_loader = DataLoader()
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    emission_control_start_timestep = time_horizon.year_to_timestep(
        year=emission_control_start_year, timestep=timestep
    )
    temperature_year_of_interest_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=timestep
    )

    # Define constants, uncertainties and levers
    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_control_start_timestep),
        Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", len(data_loader.REGION_LIST)),
        Constant("social_welfare_function_type", social_welfare_function_type),
        Constant("economy_type", economy_type.value),
        Constant("damage_function_type", damage_function_type.value),
        Constant("abatement_type", abatement_type.value),
        Constant(
            "temperature_year_of_interest_index", temperature_year_of_interest_index
        ),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_ensemble_members),
    ]

    # Speicify uncertainties
    model.uncertainties = [
        CategoricalParameter(
            "ssp_rcp_scenario",
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ),  # TODO should have a configuration file for optimizations
        ),  # 8 SSP-RCP scenario combinations
    ]

    # Set the model levers, which are the RBF parameters
    # These are the formula to calculate the number of centers, radii and weights

    centers_shape = (
        n_rbfs * n_inputs
    )  # centers = n_rbfs x n_inputs # radii = n_rbfs x n_inputs
    weights_shape = (
        len(data_loader.REGION_LIST) * n_rbfs
    )  # weights = n_outputs x n_rbfs

    centers_levers = []
    radii_levers = []
    weights_levers = []

    for i in range(centers_shape):
        centers_levers.append(
            RealParameter(f"center {i}", -1.0, 1.0)
        )  # TODO should have a configuration file for optimizations
        radii_levers.append(
            RealParameter(f"radii {i}", SMALL_NUMBER, 1.0)
        )  # Changed from 0 to SMALL_NUMBER to avoid division by zero in RBF calculations

    for i in range(weights_shape):
        weights_levers.append(
            RealParameter(f"weights {i}", SMALL_NUMBER, 1.0)
        )  # Probably this range determines the min and max values of the ECR

    # Set the model levers
    model.levers = centers_levers + radii_levers + weights_levers

    model.outcomes = [
        ScalarOutcome(
            "welfare",
            variable_name="welfare",
            kind=ScalarOutcome.MINIMIZE,
        ),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    reference_scenario = Scenario(
        "reference",
        ssp_rcp_scenario=reference_ssp_rcp_scenario_index,
    )

    # Add social_welfare_function.value[1] to the filename
    filename = f"{social_welfare_function.value[1]}_{nfe}_{seed}.tar.gz"
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    directory_name = os.path.join(
        datapath, f"{social_welfare_function.value[1]}_{date}_{seed}"
    )

    # Gate I/O to rank 0 to avoid race conditions and missing temp files
    rank = _mpi_rank()

    if rank == 0:
        os.makedirs(directory_name, exist_ok=True)
        convergence_metrics = [
            ArchiveLogger(
                directory_name,
                [l.name for l in model.levers],
                [o.name for o in model.outcomes],
                base_filename=filename,
            ),
            EpsilonProgress(),
        ]
    else:
        # No convergence/logging on non-master ranks
        convergence_metrics = []

    algorithm = None
    if optimizer == Optimizer.EpsNSGAII:
        algorithm = EpsNSGAII
    elif optimizer == Optimizer.BorgMOEA:
        algorithm = MMBorgMOEA

    set_ema_context(model=model, reference=reference_scenario)

    if evaluator == Evaluator.MPIEvaluator:
        with MPIEvaluator(model) as _evaluator:  # Use this for HPC
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence_metrics,
                population_size=population_size,
                algorithm=algorithm,
            )
    elif evaluator == Evaluator.MultiprocessingEvaluator:
        with MultiprocessingEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence_metrics,
                population_size=population_size,
                algorithm=algorithm,
            )
    else:

        # with MPIEvaluator(model) as evaluator:  # Use this for HPC
        with SequentialEvaluator(model) as _evaluator:  # Use this for local machine
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence_metrics,
                population_size=population_size,  # NOTE set population parameters for local machine. It is faster for testing
                algorithm=algorithm,
            )


def _mpi_rank() -> int:
    import os

    for k in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MPI_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


if __name__ == "__main__":
    seeds = [
        9845531,
        1644652,
        3569126,
        6075612,
        521475,
    ]

    config_path = "analysis/normative_uncertainty_optimization.json"  # This loads the config used in the Paper

    ema_logging.log_to_stderr(ema_logging.INFO)

    seed = seeds[4]
    random.seed(seed)
    np.random.seed(seed)
    # perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None)
    # run_optimization_adaptive(
    #     n_rbfs=4, n_inputs=2, nfe=5, swf=4, filename=None, folder=None, seed=seed
    # )
    run_optimization_adaptive(
        config_path=config_path,
        nfe=10,
        swf=0,
        seed=seed,
        datapath="./data",
        optimizer=Optimizer.BorgMOEA,  # Optimizer.EpsNSGAII,  # Optimizer.BorgMOEA,
        population_size=100,
        evaluator=Evaluator.SequentialEvaluator,
    )
