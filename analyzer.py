"""
Orchestrates a multi-objective optimization of the JUSTICE model using Borg.
- Builds the EMA Model (constants, levers, outcomes).
- Registers the model wrapper with the Borg adapter (direct evaluation).
- Defines MPI (MS/MM) subclasses that only tweak Borg’s library/settings.
- Drives EMA’s optimize(...) with convergence metrics (rank 0 only).
"""

import datetime
import json
import os
import warnings

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    CategoricalParameter,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    MPIEvaluator,
    Constant,
    Scenario,
)
from ema_workbench.em_framework.optimization import (
    ArchiveLogger,
    EpsilonProgress,
    EpsNSGAII,
)

from justice.util.EMA_model_wrapper import model_wrapper_emodps
from justice.util.data_loader import DataLoader
from justice.util.enumerations import (
    Abatement,
    DamageFunction,
    Economy,
    Evaluator,
    Optimizer,
    WelfareFunction,
)
from justice.util.model_time import TimeHorizon

from borg_platypus_adapter import (
    BorgMOEA,
    set_ema_context,
    _ArchiveView,
    _AlgorithmStub,
)
from platypus import Solution

SMALL_NUMBER = 1e-9
warnings.filterwarnings("ignore")


def _mpi_rank() -> int:
    """Determine MPI/Slurm rank if present, else 0."""
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MPI_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


class MSBorgMOEA(BorgMOEA):
    """Master–Slave Borg (islands = 1)."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgms.so",
            solve_settings={},
            seed=None,  # keep Borg's internal RNG
            direct_evaluation=True,  # use the evaluation function registered in context
            **kwargs,
        )

    def run(self, max_evaluations: int):
        """Run Borg in master–slave MPI mode (islands = 1)."""
        from borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)
        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        if _mpi_rank() == 0:
            print(f"[MSBorgMOEA] epsilons = {self.epsilons}", flush=True)

        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=1,
                maxEvaluations=int(max_evaluations),
                runtime=b"ms.runtime",
            )
        finally:
            Configuration.stopMPI()

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

        # Update stub
        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        return


class MMBorgMOEA(BorgMOEA):
    """Multi-Master Borg (requires islands*(workers+1)+1 MPI ranks)."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgmm.so",
            solve_settings={"islands": islands},
            seed=None,
            direct_evaluation=True,
            **kwargs,
        )
        self._islands = islands

    def run(self, max_evaluations: int):
        """Run Borg in multi-master MPI mode."""
        from borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)
        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        if _mpi_rank() == 0:
            print(
                f"[MMBorgMOEA] epsilons = {self.epsilons}, islands = {self._islands}",
                flush=True,
            )

        runtime_dir = os.environ.get("BORG_RUNTIME_DIR")
        if runtime_dir is None:
            raise RuntimeError(
                "BORG_RUNTIME_DIR is not set. "
                "Set this environment variable to the desired output folder before running the optimizer."
            )

        runtime_template = os.path.join(runtime_dir, "mm_%d.runtime").encode("utf-8")

        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=self._islands,
                maxEvaluations=int(max_evaluations),
                runtime=runtime_template,
            )
        finally:
            Configuration.stopMPI()

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

    def step(self):
        return


def run_optimization_adaptive(
    config_path,
    nfe=None,
    population_size=100,
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
    """Set up the EMA Model, register evaluation context, and run optimize()."""
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
    reference_index = config["reference_ssp_rcp_scenario_index"]
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    social_welfare_function = WelfareFunction.from_index(swf)
    swf_type = social_welfare_function.value[0]

    model = Model("JUSTICE", function=model_wrapper_emodps)

    data_loader = DataLoader()
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    emission_start_ts = time_horizon.year_to_timestep(
        year=emission_control_start_year, timestep=timestep
    )
    temperature_year_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=timestep
    )

    # Set constants (available inside the callback)
    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_start_ts),
        Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", len(data_loader.REGION_LIST)),
        Constant("social_welfare_function_type", swf_type),
        Constant("economy_type", economy_type.value),
        Constant("damage_function_type", damage_function_type.value),
        Constant("abatement_type", abatement_type.value),
        Constant("temperature_year_of_interest_index", temperature_year_index),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_members),
    ]

    # Single categorical uncertainty
    model.uncertainties = [CategoricalParameter("ssp_rcp_scenario", tuple(range(8)))]

    centers_shape = n_rbfs * n_inputs
    weights_shape = len(data_loader.REGION_LIST) * n_rbfs

    centers = [RealParameter(f"center {i}", -1.0, 1.0) for i in range(centers_shape)]
    radii = [
        RealParameter(f"radii {i}", SMALL_NUMBER, 1.0) for i in range(centers_shape)
    ]
    weights = [
        RealParameter(f"weights {i}", SMALL_NUMBER, 1.0) for i in range(weights_shape)
    ]
    model.levers = centers + radii + weights

    # Two objectives (welfare, fraction above threshold)
    model.outcomes = [
        ScalarOutcome("welfare", variable_name="welfare", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    reference_scenario = Scenario("reference", ssp_rcp_scenario=reference_index)

    filename = f"{social_welfare_function.value[1]}_{nfe}_{seed}.tar.gz"
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    directory_name = os.path.abspath(
        os.path.join(datapath, f"{social_welfare_function.value[1]}_{timestamp}_{seed}")
    )

    # Ensure every rank writes mm_*.runtime into this run directory
    os.environ["BORG_RUNTIME_DIR"] = directory_name
    os.makedirs(directory_name, exist_ok=True)

    rank = _mpi_rank()
    if rank == 0:
        convergence = [
            ArchiveLogger(
                directory_name,
                [lever.name for lever in model.levers],
                [outcome.name for outcome in model.outcomes],
                base_filename=filename,
            ),
            EpsilonProgress(),
        ]
    else:
        convergence = []

    if optimizer == Optimizer.EpsNSGAII:
        algorithm_class = EpsNSGAII
    elif optimizer == Optimizer.BorgMOEA:
        algorithm_class = MMBorgMOEA
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Register context for the Borg adapter (model, scenario, evaluation function)
    set_ema_context(
        model=model,
        reference=reference_scenario,
        evaluation=model_wrapper_emodps,
        reference_index=reference_index,
    )

    if evaluator == Evaluator.MPIEvaluator:
        with MPIEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence,
                population_size=population_size,
                algorithm=algorithm_class,
            )
    elif evaluator == Evaluator.MultiprocessingEvaluator:
        with MultiprocessingEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence,
                population_size=population_size,
                algorithm=algorithm_class,
            )
    else:
        with SequentialEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                convergence=convergence,
                population_size=population_size,
                algorithm=algorithm_class,
            )
    return results


if __name__ == "__main__":
    config_path = "analysis/normative_uncertainty_optimization.json"

    if _mpi_rank() != 0:
        ema_logging.log_to_stderr(ema_logging.CRITICAL)  # silence non-master ranks
    else:
        ema_logging.log_to_stderr(ema_logging.INFO)

    run_optimization_adaptive(
        config_path=config_path,
        nfe=10,
        swf=0,
        seed=None,
        datapath="./data",
        optimizer=Optimizer.BorgMOEA,
        population_size=100,
        evaluator=Evaluator.SequentialEvaluator,
    )
