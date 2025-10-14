import datetime
import json
import os
import random
import warnings

import numpy as np
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
from platypus import Solution, Integer, Binary

from borg_platypus_adapter import (
    BorgMOEA,
    set_ema_context,
    _ArchiveView,
    _AlgorithmStub,
    _EMA_CONTEXT,
)

SMALL_NUMBER = 1e-9  # Used to avoid division by zero in RBF calculations
warnings.filterwarnings("ignore")


def _mpi_rank() -> int:
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MPI_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


class MSBorgMOEA(BorgMOEA):
    """Master–Slave Borg: requires MPI and >= 2 ranks."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgms.so",
            solve_settings={},
            seed=None,
            **kwargs,
        )

    def _make_callback(self):
        problem = self.problem
        nconstr = getattr(problem, "nconstr", 0)

        def cb(*x):
            casted_vars = []
            for val, t in zip(x, problem.types):
                if isinstance(t, Integer):
                    lo, hi = t.min_value, t.max_value
                    v = int(round(val))
                    if lo is not None:
                        v = max(lo, v)
                    if hi is not None:
                        v = min(hi, v)
                    casted_vars.append(v)
                elif isinstance(t, Binary):
                    casted_vars.append(1 if val >= 0.5 else 0)
                else:
                    lo = getattr(t, "min_value", None)
                    hi = getattr(t, "max_value", None)
                    v = float(val)
                    if lo is not None:
                        v = max(lo, v)
                    if hi is not None:
                        v = min(hi, v)
                    casted_vars.append(v)

            mdl = _EMA_CONTEXT.get("model")
            ref = _EMA_CONTEXT.get("reference")
            if mdl is None:
                raise RuntimeError(
                    "EMA context model not set. Call set_ema_context(model, reference)."
                )

            base_kwargs = {c.name: c.value for c in getattr(mdl, "constants", [])}

            if ref is not None and hasattr(ref, "ssp_rcp_scenario"):
                base_kwargs["ssp_rcp_scenario"] = ref.ssp_rcp_scenario
            else:
                idx = _EMA_CONTEXT.get("reference_ssp_rcp_scenario_index")
                if idx is not None:
                    base_kwargs["ssp_rcp_scenario"] = idx

            if self._lever_names is None or len(self._lever_names) != len(casted_vars):
                raise RuntimeError(
                    f"Lever count mismatch: nvars={len(casted_vars)} "
                    f"vs lever_names={len(self._lever_names) if self._lever_names else None}"
                )

            lever_map = {name: val for name, val in zip(self._lever_names, casted_vars)}
            kwargs = {**base_kwargs, **lever_map}

            out = model_wrapper_emodps(**kwargs)

            if isinstance(out, tuple):
                w = out[0] if len(out) > 0 else 0.0
                frac = out[1] if len(out) > 1 else 0.0
            elif isinstance(out, dict):
                names = self._outcome_names or list(out.keys())
                w = out.get(names[0], 0.0)
                frac = out.get(names[1], 0.0) if len(names) > 1 else 0.0
            else:
                w, frac = out, 0.0

            w = (
                float(np.asarray(w).mean())
                if isinstance(w, (list, tuple, np.ndarray))
                else float(w)
            )
            frac = (
                float(np.asarray(frac).mean())
                if isinstance(frac, (list, tuple, np.ndarray))
                else float(frac)
            )

            objs = [w, frac]
            if nconstr:
                return (objs, [0.0] * nconstr)
            else:
                return objs

        return cb

    def run(self, max_evaluations: int):
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

        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        return


class MMBorgMOEA(BorgMOEA):
    """Multi-Master Borg: requires P = islands*(K+1)+1 MPI ranks."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        import os

        kwargs.pop("seed", None)
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            borg_library_path="./libborgmm.so",
            solve_settings={"islands": islands},
            seed=None,
            **kwargs,
        )
        self._islands = islands

    def _make_callback(self):
        problem = self.problem
        nconstr = getattr(problem, "nconstr", 0)

        def cb(*x):
            casted_vars = []
            for val, t in zip(x, problem.types):
                if isinstance(t, Integer):
                    lo, hi = t.min_value, t.max_value
                    v = int(round(val))
                    if lo is not None:
                        v = max(lo, v)
                    if hi is not None:
                        v = min(hi, v)
                    casted_vars.append(v)
                elif isinstance(t, Binary):
                    casted_vars.append(1 if val >= 0.5 else 0)
                else:
                    lo = getattr(t, "min_value", None)
                    hi = getattr(t, "max_value", None)
                    v = float(val)
                    if lo is not None:
                        v = max(lo, v)
                    if hi is not None:
                        v = min(hi, v)
                    casted_vars.append(v)

            mdl = _EMA_CONTEXT.get("model")
            ref = _EMA_CONTEXT.get("reference")
            if mdl is None:
                raise RuntimeError(
                    "EMA context model not set. Call set_ema_context(model, reference)."
                )

            base_kwargs = {c.name: c.value for c in getattr(mdl, "constants", [])}

            if ref is not None and hasattr(ref, "ssp_rcp_scenario"):
                base_kwargs["ssp_rcp_scenario"] = ref.ssp_rcp_scenario
            else:
                idx = _EMA_CONTEXT.get("reference_ssp_rcp_scenario_index")
                if idx is not None:
                    base_kwargs["ssp_rcp_scenario"] = idx

            if self._lever_names is None or len(self._lever_names) != len(casted_vars):
                raise RuntimeError(
                    f"Lever count mismatch: nvars={len(casted_vars)} "
                    f"vs lever_names={len(self._lever_names) if self._lever_names else None}"
                )
            lever_map = {name: val for name, val in zip(self._lever_names, casted_vars)}
            kwargs = {**base_kwargs, **lever_map}

            out = model_wrapper_emodps(**kwargs)

            if isinstance(out, tuple):
                w = out[0] if len(out) > 0 else 0.0
                frac = out[1] if len(out) > 1 else 0.0
            elif isinstance(out, dict):
                names = self._outcome_names or list(out.keys())
                w = out.get(names[0], 0.0)
                frac = out.get(names[1], 0.0) if len(names) > 1 else 0.0
            else:
                w, frac = out, 0.0

            w = (
                float(np.asarray(w).mean())
                if isinstance(w, (list, tuple, np.ndarray))
                else float(w)
            )
            frac = (
                float(np.asarray(frac).mean())
                if isinstance(frac, (list, tuple, np.ndarray))
                else float(frac)
            )

            objs = [w, frac]
            if nconstr:
                return (objs, [0.0] * nconstr)
            else:
                return objs

        return cb

    def run(self, max_evaluations: int):
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

        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=self._islands,
                maxEvaluations=int(max_evaluations),
                runtime=b"mm_%d.runtime",
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
    emission_control_start_ts = time_horizon.year_to_timestep(
        year=emission_control_start_year, timestep=timestep
    )
    temperature_year_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=timestep
    )

    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_control_start_ts),
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

    model.uncertainties = [
        CategoricalParameter("ssp_rcp_scenario", tuple(range(8))),
    ]

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

    model.outcomes = [
        ScalarOutcome("welfare", variable_name="welfare", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    reference_scenario = Scenario(
        "reference", ssp_rcp_scenario=reference_ssp_rcp_scenario_index
    )

    filename = f"{social_welfare_function.value[1]}_{nfe}_{seed}.tar.gz"
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    directory_name = os.path.join(
        datapath, f"{social_welfare_function.value[1]}_{timestamp}_{seed}"
    )

    rank = _mpi_rank()
    if rank == 0:
        os.makedirs(directory_name, exist_ok=True)
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

    set_ema_context(model=model, reference=reference_scenario)
    _EMA_CONTEXT["reference_ssp_rcp_scenario_index"] = reference_ssp_rcp_scenario_index

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
    return results  # optional; many callers ignore the return value


if __name__ == "__main__":

    config_path = "analysis/normative_uncertainty_optimization.json"

    if _mpi_rank() != 0:
        ema_logging.log_to_stderr(ema_logging.CRITICAL)
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
