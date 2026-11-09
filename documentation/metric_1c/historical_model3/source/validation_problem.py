import numpy as np
from pymoo.core.problem import Problem


class ValidationLayoutProblem(Problem):
    """
    Two-objective layout optimisation problem for optimiser validation.

    Objective 1:
        minimise 1 - normalised AEP

    Objective 2:
        minimise normalised artificial collision risk

    This file is only for the validation/sanity-check phase.
    """

    def __init__(
        self,
        scenario,
        aep_model,
        risk_model,
        aep_min,
        aep_max,
        risk_min,
        risk_max,
    ):
        self.scenario = scenario
        self.aep_model = aep_model
        self.risk_model = risk_model

        self.site = scenario.site
        self.turbine = scenario.turbine

        self.n_turbines = self.site.n_turbines
        self.width = self.site.width_m
        self.height = self.site.height_m

        self.min_spacing_m = (
            self.site.min_spacing_D * self.turbine.rotor_diameter_m
        )

        self.aep_min = float(aep_min)
        self.aep_max = float(aep_max)
        self.risk_min = float(risk_min)
        self.risk_max = float(risk_max)

        n_var = 2 * self.n_turbines

        # One spacing constraint for every turbine pair
        n_constraints = self.n_turbines * (self.n_turbines - 1) // 2

        # Lower and upper bounds for every x,y coordinate
        xl = np.tile([0.0, 0.0], self.n_turbines)
        xu = np.tile([self.width, self.height], self.n_turbines)

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=n_constraints,
            xl=xl,
            xu=xu,
        )

    def _normalise_aep(self, aep):
        denom = self.aep_max - self.aep_min

        if abs(denom) < 1e-9:
            return 0.0

        return (aep - self.aep_min) / denom

    def _normalise_risk(self, risk):
        denom = self.risk_max - self.risk_min

        if abs(denom) < 1e-9:
            return 0.0

        return (risk - self.risk_min) / denom

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float)

        F = np.zeros((X.shape[0], 2))
        G = np.zeros((X.shape[0], self.n_ieq_constr))

        for p in range(X.shape[0]):
            xy = X[p].reshape(self.n_turbines, 2)

            raw_aep = self.aep_model.evaluate(xy)
            raw_risk = self.risk_model.collisions(xy)

            aep_norm = self._normalise_aep(raw_aep)
            risk_norm = self._normalise_risk(raw_risk)

            # Objective 1: maximise AEP by minimising 1 - normalised AEP
            F[p, 0] = 1.0 - aep_norm

            # Objective 2: minimise collision risk
            F[p, 1] = risk_norm

            # Spacing constraints: feasible when G <= 0
            c = 0

            for i in range(self.n_turbines):
                for j in range(i + 1, self.n_turbines):
                    distance = np.linalg.norm(xy[i] - xy[j])
                    G[p, c] = self.min_spacing_m - distance
                    c += 1

        out["F"] = F
        out["G"] = G