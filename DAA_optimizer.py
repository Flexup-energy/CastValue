import numpy as np
import pyomo.environ as pyo

class Optimizer:
    def __init__(self, solverpath_exe):
        self.solverpath_exe = solverpath_exe

    def set_glpk_solver(self):
        return pyo.SolverFactory("glpk", executable=self.solverpath_exe)

    def optimize_daa(self, n_cycles, energy_cap, power_cap, daa_price_vector,
                     initial_soc=0.0, rte=0.85):
        """
        Optimizes DAA charge/discharge schedule for a 96-interval (15-min) price vector.

        Energy balance uses charge-side RTE losses:
          SOC[q+1] = SOC[q] + rte * pf * cha[q] - pf * dis[q]
        i.e. 1 MWh drawn from grid stores rte MWh in battery; 1 MWh delivered
        to grid draws 1 MWh from battery. Volume limit is defined on the
        battery-side (discharge delivery); charge-side grid volume is capped
        at volume_limit/rte to allow a full discharge cycle.

        Args:
            n_cycles: max discharge cycles per day (battery-side)
            energy_cap: battery energy capacity (MWh)
            power_cap: battery power capacity (MW)
            daa_price_vector: list of 96 forecast prices (EUR/MWh)
            initial_soc: starting SOC in MWh (for rolling SOC across days)
            rte: round-trip efficiency (default 0.85)

        Returns:
            (soc, cha, dis) — lists of 96 values each
        """
        model = pyo.ConcreteModel()
        model.Q = pyo.RangeSet(1, 96)
        model.Q_plus_1 = pyo.RangeSet(1, 97)
        volume_limit = energy_cap * n_cycles
        pf = power_cap / 4.0

        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        model.cha = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.cons = pyo.ConstraintList()
        model.cons.add(model.soc[1] == initial_soc)
        model.cons.add(model.soc[97] >= 0)

        for q in model.Q:
            model.cons.add(model.soc[q] >= 0)
            model.cons.add(model.soc[q] <= energy_cap)
            model.cons.add(model.soc[q+1] == model.soc[q] + rte * pf * model.cha[q] - pf * model.dis[q])

        model.cons.add(sum(model.cha[q] * pf for q in model.Q) <= volume_limit / rte)
        model.cons.add(sum(model.dis[q] * pf for q in model.Q) <= volume_limit)

        model.obj = pyo.Objective(
            expr=sum(pf * daa_price_vector[q-1] * (model.dis[q] - model.cha[q]) for q in model.Q),
            sense=pyo.maximize
        )

        solver = self.set_glpk_solver()
        solver.solve(model, timelimit=5)

        return (
            [model.soc[q].value for q in model.Q],
            [model.cha[q].value for q in model.Q],
            [model.dis[q].value for q in model.Q]
        )

    def run_day(self, forecasts, actuals, power_cap, energy_cap, n_cycles,
                rte=0.85, initial_soc=0.0, perfect_foresight=False):
        """
        Paired opportunity-cost bidding with RTE and volume-balanced rejection.

        1. Runs LP on forecast prices to get optimal charge/discharge schedule
        2. Derives bid prices from the forecast spread, adjusted for RTE:
           - Charge bid  = avg_forecast_discharge_price × RTE
           - Discharge ask = avg_forecast_charge_price / RTE
        3. Checks each MTU against actual clearing price (binary OTM fill)
        4. Enforces paired rejection: if cleared charge and discharge volumes
           differ, rejects worst-margin MTUs on the oversubscribed side until
           volumes match. Guarantees end_soc == initial_soc (no carryover,
           no position imbalance).

        If perfect_foresight=True, the OTM filter and paired rejection are
        skipped: with perfect forecast the trader bids at the clearing price,
        so every LP-scheduled MTU fills. RTE losses are still captured via
        the LP energy-balance constraint.

        Args:
            forecasts: dict with key "daa" → list of 96 forecast prices
            actuals:   dict with key "daa" → list of 96 actual prices
            power_cap: MW
            energy_cap: MWh
            n_cycles: max cycles per day
            rte: round-trip efficiency (default 0.85)
            initial_soc: starting SOC in MWh
            perfect_foresight: if True, all bids clear (use only for the
                perfect-foresight benchmark scenario)

        Returns dict with:
            profit_total        — realized profit after OTM + paired rejection
            bid_prices          — bid/ask price per MTU
            fill_mask           — 1 if filled, 0 if rejected (post-balancing)
            fill_rate           — fraction of active MTUs filled
            avg_charge_price    — volume-weighted avg forecast charge price
            avg_discharge_price — volume-weighted avg forecast discharge price
            soc                 — realized SOC trajectory (96 values)
            end_soc             — equals initial_soc by construction
            cha / dis           — realized charge/discharge rates
            cha_planned / dis_planned — LP-optimized rates before OTM
        """
        soc_planned, cha_planned, dis_planned = self.optimize_daa(
            n_cycles=n_cycles,
            energy_cap=energy_cap,
            power_cap=power_cap,
            daa_price_vector=forecasts["daa"],
            initial_soc=initial_soc,
            rte=rte,
        )

        forecast = np.array(forecasts["daa"])
        actual   = np.array(actuals["daa"])
        cha_plan = np.array(cha_planned)
        dis_plan = np.array(dis_planned)
        pf = power_cap / 4.0

        # Volume-weighted average forecast prices for each side
        cha_energy = cha_plan * pf
        dis_energy = dis_plan * pf

        total_cha = cha_energy.sum()
        total_dis = dis_energy.sum()

        avg_charge_price = float(np.sum(forecast * cha_energy) / total_cha) if total_cha > 1e-9 else 0.0
        avg_discharge_price = float(np.sum(forecast * dis_energy) / total_dis) if total_dis > 1e-9 else 0.0

        # Bid prices adjusted for RTE (or clearing-price bids under perfect foresight)
        bid_price = np.copy(actual) if perfect_foresight else np.copy(forecast)
        if not perfect_foresight:
            for q in range(96):
                if cha_plan[q] > 1e-6:
                    bid_price[q] = avg_discharge_price * rte
                elif dis_plan[q] > 1e-6:
                    bid_price[q] = avg_charge_price / rte

        # OTM check: binary per MTU (skipped under perfect foresight — all fill)
        cha_accepted = np.zeros(96)
        dis_accepted = np.zeros(96)

        if perfect_foresight:
            cha_accepted = cha_plan.copy()
            dis_accepted = dis_plan.copy()
        else:
            for q in range(96):
                if cha_plan[q] > 1e-6 and bid_price[q] >= actual[q]:
                    cha_accepted[q] = cha_plan[q]
                elif dis_plan[q] > 1e-6 and bid_price[q] <= actual[q]:
                    dis_accepted[q] = dis_plan[q]

        # Paired rejection: match cleared volumes on the battery side
        # (cha_stored = dis_drawn → end_soc = initial_soc by construction).
        # Grid-side charge MWh convert to battery MWh at rte; grid-side
        # discharge is 1:1 with battery drawdown.
        cha_final = cha_accepted.copy()
        dis_final = dis_accepted.copy()
        V_cha_stored = cha_final.sum() * pf * rte
        V_dis_drawn = dis_final.sum() * pf

        if V_cha_stored > V_dis_drawn + 1e-9:
            excess = V_cha_stored - V_dis_drawn  # battery-side excess
            cleared = np.where(cha_final > 1e-9)[0]
            # worst charge = highest actual price paid
            order = cleared[np.argsort(-actual[cleared])]
            for q in order:
                if excess <= 1e-9:
                    break
                e_stored = cha_final[q] * pf * rte
                if excess >= e_stored - 1e-12:
                    cha_final[q] = 0.0
                    excess -= e_stored
                else:
                    cha_final[q] -= excess / (pf * rte)
                    excess = 0.0
        elif V_dis_drawn > V_cha_stored + 1e-9:
            excess = V_dis_drawn - V_cha_stored
            cleared = np.where(dis_final > 1e-9)[0]
            # worst discharge = lowest actual price received
            order = cleared[np.argsort(actual[cleared])]
            for q in order:
                if excess <= 1e-9:
                    break
                e_drawn = dis_final[q] * pf
                if excess >= e_drawn - 1e-12:
                    dis_final[q] = 0.0
                    excess -= e_drawn
                else:
                    dis_final[q] -= excess / pf
                    excess = 0.0

        # Realized SOC trajectory with charge-side RTE loss
        soc_real = np.zeros(97)
        soc_real[0] = initial_soc
        for q in range(96):
            soc_real[q + 1] = soc_real[q] + rte * cha_final[q] * pf - dis_final[q] * pf

        fill_mask = ((cha_final > 1e-6) | (dis_final > 1e-6)).astype(float)

        profit = float(np.sum((dis_final - cha_final) * actual) * pf)
        end_soc = float(soc_real[96])

        active = (cha_plan > 1e-6) | (dis_plan > 1e-6)
        fill_rate = float(np.mean(fill_mask[active])) if active.any() else 1.0

        return {
            "profit_total":       profit,
            "bid_prices":         bid_price.tolist(),
            "fill_mask":          fill_mask.tolist(),
            "fill_rate":          fill_rate,
            "avg_charge_price":   avg_charge_price,
            "avg_discharge_price": avg_discharge_price,
            "soc":                soc_real[:96].tolist(),
            "end_soc":            end_soc,
            "cha":                cha_final.tolist(),
            "dis":                dis_final.tolist(),
            "cha_planned":        cha_plan.tolist(),
            "dis_planned":        dis_plan.tolist(),
        }
