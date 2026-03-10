import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po

class Optimizer:
    def __init__(self, solverpath_exe):
        self.solverpath_exe = solverpath_exe

    def set_glpk_solver(self):
        return pyo.SolverFactory("glpk", executable=self.solverpath_exe)

    def step1_optimize_daa(self, n_cycles: int, energy_cap: int, power_cap: int, daa_price_vector: list, resolution: str = '1h'):
        """
        Optimizes DAA positions (hourly blocks) for a 96-interval price vector.
        """
        model = pyo.ConcreteModel()
        model.H = pyo.RangeSet(0, 23)   # Hours
        model.Q = pyo.RangeSet(1, 96)   # Quarters
        model.Q_plus_1 = pyo.RangeSet(1, 97)
        volume_limit = energy_cap * n_cycles

        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        model.cha_daa = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis_daa = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Constraints
        model.cons = pyo.ConstraintList()
        model.cons.add(model.soc[1] == 0)
        model.cons.add(model.soc[97] == 0)
        
        for q in model.Q:
            model.cons.add(model.soc[q] >= 0)
            model.cons.add(model.soc[q] <= energy_cap)
            model.cons.add(model.soc[q+1] == model.soc[q] + (power_cap/4 * model.cha_daa[q]) - (power_cap/4 * model.dis_daa[q]))

        model.cons.add(sum(model.cha_daa[q] * power_cap/4 for q in model.Q) <= volume_limit)
        model.cons.add(sum(model.dis_daa[q] * power_cap/4 for q in model.Q) <= volume_limit)

        # Force Hourly Blocks only if resolution is '1h' (Jan - Sept)
        # If resolution is '15m' (Oct - Dec), we skip this, allowing flexibility
        if resolution == '1h':
            for h in model.H:
                for q in range(2, 5):
                    model.cons.add(model.cha_daa[4*h + 1] == model.cha_daa[4*h + q])
                    model.cons.add(model.dis_daa[4*h + 1] == model.dis_daa[4*h + q])

        model.obj = pyo.Objective(
            expr=sum(power_cap/4 * daa_price_vector[q-1] * (model.dis_daa[q] - model.cha_daa[q]) for q in model.Q),
            sense=pyo.maximize
        )

        solver = self.set_glpk_solver()
        solver.solve(model, timelimit=5)

        return (
            [model.soc[q].value for q in model.Q],
            [model.cha_daa[q].value for q in model.Q],
            [model.dis_daa[q].value for q in model.Q]
        )

    def run_day(self, forecasts, actuals, power_cap, energy_cap, n_cycles, resolution='1h'):
        soc, cha, dis = self.step1_optimize_daa(
            n_cycles=n_cycles,
            energy_cap=energy_cap,
            power_cap=power_cap,
            daa_price_vector=forecasts["daa"],
            resolution=resolution
        )

        power_factor = power_cap / 4.0
        profit_daa = np.sum((np.array(dis) - np.array(cha)) * np.array(actuals["daa"])) * power_factor

        return {"profit_total": float(profit_daa)}