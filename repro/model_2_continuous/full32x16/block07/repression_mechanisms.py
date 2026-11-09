"""Fixed-fit RNA Repressor models: model_1 and model_2, with exact sensitivities.

No parameters are fitted here. Loading scales the target contribution in the
shared resource denominator in every translation branch.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent


@dataclass
class Parameters:
    mechanism: str
    beta: float
    extra: float
    omega: float
    e: float
    off: float
    d: float
    kG: float
    kR: float
    gR: float
    gG: float
    R: float
    M0: float
    r0: float
    kbase: float

    @property
    def h(self):
        return self.beta + self.extra + self.off

    @property
    def s(self):
        return 1 - self.omega

    @property
    def vg(self):
        return self.gG * self.R / self.kG


def load(mechanism="model_2"):
    fit = json.loads((ROOT / "committed_fit.json").read_text())
    base = json.loads((ROOT / "model_config.json").read_text())
    raw = base["params"] | fit["parameters"]["fitted_shared"]
    assert mechanism in ["model_1", "model_2"]
    suffix = "model_1" if mechanism == "model_1" else "model_2"
    length = raw["length_" + suffix]
    gR, gG = raw["gamma"] / length, raw["gamma"] / raw["length_output"]
    kG = (gG + raw["kappa_minus"]) / raw["kappa_plus"] / (raw["length_output"] / raw["ribo_footprint_size"])
    kR = (gR + raw["kappa_minus"]) / raw["kappa_plus"] / (length / raw["ribo_footprint_size"])
    ic = base["initial_conditions_RNA"]
    p = Parameters(mechanism, raw["beta"], raw["beta"] * raw["lambda_beta"] if mechanism == "model_2" else 0.,
                   raw["theta"] if mechanism == "model_2" else 0., raw["eta_plus_" + suffix], raw["eta_minus_" + suffix],
                   raw["delta"] + raw["mu"], kG, kR, gR, gG, raw["total_rtl"],
                   ic["RNA_Output"], ic["RNA_Repressor"], ic["RNA_Burden"])
    return p, fit, base


def delivery(q, cell="reference", mechanism="model_2", modality="RNA"):
    p, fit, base = load(mechanism)
    nu = fit["parameters"]["nuisance_per_arm"][f"{cell}_{mechanism}_{modality}"]
    initial = base[f"initial_conditions_{modality}"]["RNA_Burden" if modality == "RNA" else "DNA_Burden"]
    return initial * nu["kappa_burden_circuit"] * (np.asarray(q) / 20) ** nu["p_dose_circuit"]


def inputs(t, k0, p):
    decay = np.exp(-p.beta * t)
    rr, b = p.r0 / p.kR * decay, k0 / p.kG * decay
    return 1 + rr + b, b, p.gR * p.R * rr


def rhs(t, x, k0, eps, p):
    M, c, W, Y, C, n, P, H, Q, S = x
    m, Z = M - c, W - c
    a = M - p.s * c
    B, b, F = inputs(t, k0, p)
    D = B + eps * a / p.kG
    U = eps * F / (p.kG * D ** 2)
    bind = p.e * Z * m - p.h * c
    V = p.vg * a / D
    # n=M_logdose, P=-c_logdose, H=-W_logdose. P need not be nonnegative.
    Vq = p.vg * (B * (n + p.s * P) - a * b) / D ** 2
    return np.array([
        -p.beta * M - p.extra * c,
        bind,
        F / D - p.d * Z,
        V - p.d * Y,
        Y,
        -p.beta * n + p.extra * P,
        -p.e * Z * n - (p.e * Z + p.h + p.e * m) * P + p.e * m * H,
        U * n + (p.d + p.s * U) * P - p.d * H + F * b / D ** 2,
        Vq - p.d * Q,
        Q,
    ])


def solve(k0, eps, mechanism="model_2", T=48., rtol=2e-11):
    p, _, _ = load(mechanism)
    x0 = np.zeros(10)
    x0[0] = p.M0
    sol = solve_ivp(lambda t, x: rhs(t, x, k0, eps, p), (0, T), x0,
                    method="Radau", rtol=rtol, atol=rtol / 10, dense_output=True)
    assert sol.success, sol.message
    return sol


class Comparable:
    """Adapt to the archived waveform-metric generator's eight-state order."""
    def __init__(self, sol):
        self.solution = sol

    def sol(self, t):
        x = self.solution.sol(t)
        return x[[0, 1, 2, 5, 3, 4, 8, 9]]


def archive_solve(q, eps, mechanism="model_2", cell="reference", modality="RNA", T=48., rtol=2e-11):
    p, fit, _ = load(mechanism)
    path = ROOT / "source/simulator/model.py"
    spec = importlib.util.spec_from_file_location("archived_repression_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Repressor_iFFL(T, modality, "branch_1" if mechanism == "model_1" else "branch_2")
    model.update_parameters(fit["parameters"]["fitted_shared"])
    model._adjust_parameters()
    model.update_initial_conditions({("RNA_Burden" if modality == "RNA" else "DNA_Burden"): float(delivery(q, cell, mechanism, modality))})
    model._set_x0()
    pars = model.params
    n = len(model.x0)
    yi = 4 if modality == "RNA" else 7
    zi = 5 if modality == "RNA" else 8
    ki = 6 if modality == "RNA" else 9
    offset = 0 if modality == "RNA" else 3

    def function(t, x):
        dx = model.model(t, x[:n])
        m, c, r, k = x[offset:offset + 4]
        a = m + pars["theta"] * c
        B = 1 + r / pars["kappa_repression"] + k / pars["kappa_burden"]
        R1 = pars["total_rtl"] / (B + a / pars["kappa_output"])
        Re = pars["total_rtl"] / (B + eps * a / pars["kappa_output"])
        dx[yi] += pars["gamma_output"] * a / pars["kappa_output"] * (Re - R1)
        dx[zi] += pars["gamma_repression"] * r / pars["kappa_repression"] * (Re - R1)
        dx[ki] += pars["gamma_burden"] * k / pars["kappa_burden"] * (Re - R1)
        return np.r_[dx, x[yi]]

    sol = solve_ivp(function, (0, T), np.r_[model.x0, 0.], method="Radau",
                    rtol=rtol, atol=rtol / 10, dense_output=True)
    assert sol.success
    return sol
