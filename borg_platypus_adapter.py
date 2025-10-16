"""
Adapter that wraps the C Borg implementation (via borg.py) so it can be used as
a Platypus Algorithm inside EMA-Workbench.

Highlights
----------
- Stores EMA model/Scenario/lever names in a shared context.
- Builds a callback that either evaluates solutions via Platypus’ Problem
  (default) or directly via a user-specified evaluation function (direct mode).
- Presents minimal archive stubs so EMA convergence metrics keep working.
"""

from typing import List, Optional, Dict, Any, Iterable

import numpy as np
from platypus import Algorithm, Problem, Solution, Real, Integer, Binary

# Shared EMA context; set once from user code
_EMA_CONTEXT: Dict[str, Any] = {
    "model": None,
    "reference": None,
    "lever_names": None,
    "outcome_names": None,
    "evaluation": None,
    "reference_index": None,
}


def set_ema_context(
    model=None,
    reference=None,
    lever_names=None,
    outcome_names=None,
    evaluation=None,
    reference_index=None,
) -> None:
    """
    Populate the EMA context so the Borg callback can fetch constants, levers,
    outcomes, evaluation function, and reference scenario.

    Parameters
    ----------
    model : ema_workbench.Model
        Model object (used for constants/levers/outcomes).
    reference : ema_workbench.Scenario
        Scenario used as the reference case.
    lever_names : list[str], optional
        Lever names (if not provided, tries to infer from model).
    outcome_names : list[str], optional
        Outcome names (if not provided, tries to infer from model).
    evaluation : callable, optional
        Function to evaluate a single policy (e.g., model_wrapper_emodps).
        Signature should be ``evaluation(**kwargs) -> Tuple[float, float]`` or similar.
    reference_index : int, optional
        Optional scenario index so the callback can set `ssp_rcp_scenario`.
    """
    _EMA_CONTEXT["model"] = model
    _EMA_CONTEXT["reference"] = reference
    _EMA_CONTEXT["evaluation"] = evaluation
    _EMA_CONTEXT["reference_index"] = reference_index

    if model is not None:
        try:
            if lever_names is None and hasattr(model, "levers"):
                lever_names = [lever.name for lever in model.levers]
        except Exception:
            pass
        try:
            if outcome_names is None and hasattr(model, "outcomes"):
                outcome_names = [outcome.name for outcome in model.outcomes]
        except Exception:
            pass

    _EMA_CONTEXT["lever_names"] = lever_names
    _EMA_CONTEXT["outcome_names"] = outcome_names


class _ArchiveView:
    """Minimal archive stub so EMA convergence metrics can iterate/inspect."""

    def __init__(self, solutions: List[Solution], improvements: int = 0) -> None:
        self._solutions = solutions
        self.improvements = improvements if improvements is not None else len(solutions)

    def __len__(self) -> int:
        return len(self._solutions)

    def __iter__(self) -> Iterable[Solution]:
        return iter(self._solutions)


class _AlgorithmStub:
    """Minimal algorithm stub exposing only an `.archive` attribute."""

    def __init__(self, archive: _ArchiveView) -> None:
        self.archive = archive


class BorgMOEA(Algorithm):
    """
    Base adapter for the C Borg MOEA (via borg.py) so it can ‘look like’ a
    Platypus Algorithm inside EMA-Workbench.

    Parameters
    ----------
    problem : platypus.Problem
        Platypus problem (variables, objectives, constraints).
    epsilons : List[float]
        Epsilon values for Borg’s epsilon-dominance archive.
    population_size : Optional[int]
        Initial population size (overrides Borg default).
    borg_library_path : Optional[str]
        Path to the compiled Borg shared library (e.g., libborg.so or libborgmm.so).
    seed : Optional[int]
        Seed for Borg’s RNG (use None to keep Borg’s default randomness).
    solve_settings : Optional[Dict[str, Any]]
        Additional Borg run settings (maxEvaluations is set automatically).
    direct_evaluation : bool
        When True, the callback calls the evaluation function stored in
        `_EMA_CONTEXT["evaluation"]` instead of `Problem.evaluate`.
    """

    def __init__(
        self,
        problem: Problem,
        epsilons: List[float],
        population_size: Optional[int] = None,
        borg_library_path: Optional[str] = None,
        seed: Optional[int] = None,
        solve_settings: Optional[Dict[str, Any]] = None,
        direct_evaluation: bool = False,
        **kwargs,
    ):
        super(BorgMOEA, self).__init__(problem)

        # Configuration parameters
        self.epsilons = epsilons
        self.population_size = population_size
        self.borg_library_path = borg_library_path
        self.seed = seed
        self.solve_settings = solve_settings or {}
        self.direct_evaluation = direct_evaluation

        # Extract optional EMA metadata if provided
        self.evaluator = kwargs.get("evaluator", None)
        self.reference = kwargs.get("reference", None)
        self.extra_kwargs = kwargs

        # Try inferring lever/outcome names from evaluator, else fallback to context
        self._lever_names = None
        self._outcome_names = None
        try:
            if self.evaluator is not None and hasattr(self.evaluator, "model"):
                self._lever_names = [
                    lever.name for lever in getattr(self.evaluator.model, "levers", [])
                ]
                self._outcome_names = [
                    outcome.name
                    for outcome in getattr(self.evaluator.model, "outcomes", [])
                ]
        except Exception:
            pass

        if not self._lever_names:
            self._lever_names = _EMA_CONTEXT.get("lever_names")
        if not self._outcome_names:
            self._outcome_names = _EMA_CONTEXT.get("outcome_names")
        if self.reference is None:
            self.reference = _EMA_CONTEXT.get("reference")

        # Initialize result tracking (list of Platypus solutions)
        self.result: List[Solution] = []
        self.nfe = 0

        # Provide stub archive/algorithm for EMA convergence logging
        self.archive = _ArchiveView(self.result, improvements=0)
        self.algorithm = _AlgorithmStub(self.archive)

    # ------------------------------------------------------------------
    # Helpers for the two evaluation modes
    # ------------------------------------------------------------------
    def _evaluate_with_problem(self, casted_vars: List[Any]) -> tuple:
        """Evaluate using Problem.evaluate (Platypus default)."""
        problem = self.problem
        nconstr = getattr(problem, "nconstr", 0)

        s = Solution(problem)
        s.variables = casted_vars
        problem.evaluate(s)

        if nconstr:
            cons = list(getattr(s, "constraints", []) or [])
            if len(cons) < nconstr:
                cons += [0.0] * (nconstr - len(cons))
            return (list(s.objectives), cons)
        else:
            return list(s.objectives)

    def _evaluate_directly(self, casted_vars: List[Any]) -> tuple:
        """
        Evaluate by calling the user-provided callable stored in _EMA_CONTEXT.
        Produces (objectives, constraints) in the same shape as `_evaluate_with_problem`.
        """
        eval_fn = _EMA_CONTEXT.get("evaluation")
        mdl = _EMA_CONTEXT.get("model")

        if eval_fn is None or mdl is None:
            raise RuntimeError(
                "Direct evaluation requested, but the evaluation function or model "
                "was not provided. Set evaluation=... in set_ema_context(...)."
            )

        base_kwargs = {c.name: c.value for c in getattr(mdl, "constants", [])}

        ref = _EMA_CONTEXT.get("reference")
        if ref is not None and hasattr(ref, "ssp_rcp_scenario"):
            base_kwargs["ssp_rcp_scenario"] = ref.ssp_rcp_scenario
        else:
            idx = _EMA_CONTEXT.get("reference_index")
            if idx is not None:
                base_kwargs["ssp_rcp_scenario"] = idx

        if self._lever_names is None or len(self._lever_names) != len(casted_vars):
            raise RuntimeError(
                f"Lever count mismatch: nvars={len(casted_vars)} "
                f"vs lever_names={len(self._lever_names) if self._lever_names else None}"
            )

        lever_map = {name: val for name, val in zip(self._lever_names, casted_vars)}
        kwargs = {**base_kwargs, **lever_map}

        out = eval_fn(**kwargs)

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

        problem = self.problem
        nconstr = getattr(problem, "nconstr", 0)

        if nconstr:
            return ([w, frac], [0.0] * nconstr)
        else:
            return [w, frac]

    # ------------------------------------------------------------------
    def _make_callback(self):
        """
        Build the evaluation function Borg will call. If `direct_evaluation` is set,
        we use _evaluate_directly; otherwise we fall back to Problem.evaluate.
        """
        problem = self.problem
        direct = bool(self.direct_evaluation)

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

            if direct:
                return self._evaluate_directly(casted_vars)
            else:
                return self._evaluate_with_problem(casted_vars)

        return cb

    def _set_bounds(self, borg):
        """Align Platypus bounds with Borg’s expectation."""
        bounds = []
        for t in self.problem.types:
            if isinstance(t, Binary):
                lo, hi = 0.0, 1.0
            else:
                lo = getattr(t, "min_value", None)
                hi = getattr(t, "max_value", None)
                if lo is None or hi is None:
                    raise ValueError("All variables must have finite bounds for Borg.")
            bounds.append([float(lo), float(hi)])
        borg.setBounds(*bounds)

    def run(self, max_evaluations: int):
        """
        Serial run for Borg (no MPI). Converts results to Platypus Solution objects.
        """
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

        settings = dict(self.solve_settings)
        settings["maxEvaluations"] = int(max_evaluations)
        if self.population_size is not None:
            settings.setdefault("initialPopulationSize", int(self.population_size))
            settings.setdefault("minimumPopulationSize", int(self.population_size))

        borg_result = borg.solve(settings)

        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                sol = Solution(self.problem)
                sol.variables = list(s_borg.getVariables())
                sol.objectives = list(s_borg.getObjectives())
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = settings["maxEvaluations"]
        else:
            self.nfe = 0

        # Refresh stub so EMA convergence diagnostics still work
        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        """Newer Platypus requires a concrete step() method even if run() is used."""
        return
