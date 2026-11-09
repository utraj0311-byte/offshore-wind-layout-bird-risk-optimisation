import numpy as np
from pymoo.core.problem import Problem


class RiskOnlyLayoutProblem(Problem):
    """
    Single-objective layout optimisation problem used to
    validate a coupled artificial collision-risk model.

    Objective:
        minimise total artificial collision risk.

    Constraints:
        minimum pairwise turbine spacing.

    Site boundaries are enforced through coordinate bounds.
    """

    def __init__(
        self,
        scenario,
        risk_model,
    ):
        self.scenario = scenario
        self.risk_model = risk_model

        self.site = scenario.site
        self.turbine = scenario.turbine

        self.n_turbines = self.site.n_turbines
        self.width = self.site.width_m
        self.height = self.site.height_m

        self.min_spacing_m = (
            self.site.min_spacing_D
            * self.turbine.rotor_diameter_m
        )

        n_variables = 2 * self.n_turbines

        n_constraints = (
            self.n_turbines
            * (self.n_turbines - 1)
            // 2
        )

        lower_bounds = np.tile(
            [0.0, 0.0],
            self.n_turbines,
        )

        upper_bounds = np.tile(
            [self.width, self.height],
            self.n_turbines,
        )

        super().__init__(
            n_var=n_variables,
            n_obj=1,
            n_ieq_constr=n_constraints,
            xl=lower_bounds,
            xu=upper_bounds,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        F = np.zeros((X.shape[0], 1))
        G = np.zeros(
            (X.shape[0], self.n_ieq_constr)
        )

        for population_index in range(X.shape[0]):
            xy = X[population_index].reshape(
                self.n_turbines,
                2,
            )

            F[population_index, 0] = float(
                self.risk_model.collisions(xy)
            )

            constraint_index = 0

            for i in range(self.n_turbines):
                for j in range(
                    i + 1,
                    self.n_turbines,
                ):
                    distance = np.linalg.norm(
                        xy[i] - xy[j]
                    )

                    G[
                        population_index,
                        constraint_index,
                    ] = (
                        self.min_spacing_m
                        - distance
                    )

                    constraint_index += 1

        out["F"] = F
        out["G"] = G