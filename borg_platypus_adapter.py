from typing import List, Optional, Dict, Any, Iterable
from platypus import Algorithm, Problem, Solution, Real, Integer, Binary
import os

# Global EMA context, set once by user code
_EMA_CONTEXT: Dict[str, Any] = {
    "model": None,
    "reference": None,
    "lever_names": None,
    "outcome_names": None,
}


def set_ema_context(model=None, reference=None, lever_names=None, outcome_names=None):
    """
    Provide EMA Model and reference Scenario so the adapter can:
      - Build a Policy for evaluator.perform_experiments
      - Map outcome dicts to objectives (consistent order)
    Call this once before calling evaluator.optimize(...).
    """
    _EMA_CONTEXT["model"] = model
    _EMA_CONTEXT["reference"] = reference

    # Try to extract names from model when not explicitly provided
    if model is not None:
        try:
            if lever_names is None and hasattr(model, "levers"):
                lever_names = [lv.name for lv in model.levers]
        except Exception:
            pass
        try:
            if outcome_names is None and hasattr(model, "outcomes"):
                outcome_names = [oc.name for oc in model.outcomes]
        except Exception:
            pass

    _EMA_CONTEXT["lever_names"] = lever_names
    _EMA_CONTEXT["outcome_names"] = outcome_names


class _ArchiveView:
    def __init__(self, solutions: List[Solution], improvements: int = 0):
        self._solutions = solutions
        self.improvements = improvements if improvements is not None else len(solutions)

    def __len__(self):
        return len(self._solutions)

    def __iter__(self) -> Iterable[Solution]:
        return iter(self._solutions)


class _AlgorithmStub:
    def __init__(self, archive: _ArchiveView):
        self.archive = archive


class BorgMOEA(Algorithm):
    """
    Platypus Algorithm adapter for the C Borg MOEA (using borg.py), compatible with EMA-Workbench.
    It evaluates single candidates by calling evaluator.perform_experiments(...) if problem.function is not set.
    """

    def __init__(
        self,
        problem: Problem,
        epsilons: List[float],
        population_size: Optional[int] = None,
        borg_library_path: Optional[str] = None,
        seed: Optional[int] = None,
        solve_settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super(BorgMOEA, self).__init__(problem)

        self.epsilons = epsilons
        self.population_size = population_size
        self.borg_library_path = borg_library_path
        self.seed = seed
        self.solve_settings = solve_settings or {}

        # EMA passes this; we'll use evaluator.perform_experiments
        self.evaluator = kwargs.get("evaluator", None)
        # Some EMA versions pass reference; otherwise we will take it from global context
        self.reference = kwargs.get("reference", None)

        self.extra_kwargs = kwargs

        # Names: prefer from evaluator.model if present, else from global EMA context
        self._lever_names = None
        self._outcome_names = None
        try:
            if self.evaluator is not None and hasattr(self.evaluator, "model"):
                self._lever_names = [
                    lv.name for lv in getattr(self.evaluator.model, "levers", [])
                ]
                self._outcome_names = [
                    oc.name for oc in getattr(self.evaluator.model, "outcomes", [])
                ]
        except Exception:
            pass

        # Fall back to context
        if not self._lever_names:
            self._lever_names = _EMA_CONTEXT.get("lever_names")
        if not self._outcome_names:
            self._outcome_names = _EMA_CONTEXT.get("outcome_names")
        if self.reference is None:
            self.reference = _EMA_CONTEXT.get("reference")

        # Results and accounting
        self.result: List[Solution] = []
        self.nfe = 0

        # EMA expects optimizer.archive and optimizer.algorithm.archive
        self.archive = _ArchiveView(self.result, improvements=0)
        self.algorithm = _AlgorithmStub(self.archive)

    def _make_callback(self):
        problem = self.problem
        nconstr = getattr(problem, "nconstr", 0)

        def cb(*x):
            # Cast Borg doubles into Platypus types
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

            # Evaluate via Platypus/EMA Problem
            s = Solution(problem)
            s.variables = casted_vars
            problem.evaluate(s)

            # Return objectives (and constraints if present)
            if nconstr:
                cons = list(getattr(s, "constraints", []) or [])
                if len(cons) < nconstr:
                    cons += [0.0] * (nconstr - len(cons))
                return (list(s.objectives), cons)
            else:
                return list(s.objectives)

        return cb

    def _set_bounds(self, borg):
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

        # Convert Borg results to Platypus Solutions
        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                vars_ = list(s_borg.getVariables())
                objs_ = list(s_borg.getObjectives())
                sol = Solution(self.problem)
                sol.variables = vars_
                sol.objectives = objs_
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = settings["maxEvaluations"]
        else:
            self.nfe = 0

        # Refresh archive stubs for EMA’s metrics/logging
        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        """
        Required by newer Platypus Algorithm interface.
        Not used in EMA-Workbench’s optimize path (which calls run(max_evaluations)).
        """
        return
