"""Outward-rounded model_2 RNA certificate at the unchanged joint fit.

Bilinear parameter/time polynomials are untrusted proposals. Bernstein residual
tubes cover entire parameter rectangles without a monotone-corner assumption.
A three-dimensional absolute-sensitivity comparison supplies sign/tail bounds.
"""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import certify as c
from repression_mechanisms import load

ROOT = Path(__file__).resolve().parent
I, UP, DOWN = c.I, c.UP, c.DOWN


def constants():
    raw, old, fit, base = c.load_parameters()
    v = {k: c.raw(x) for k, x in raw.items()}
    v.update(M0=c.raw(base["initial_conditions_RNA"]["RNA_Output"]),
             r0=c.raw(base["initial_conditions_RNA"]["RNA_Repressor"]))
    v["e"] = v["eta_plus_model_2"]
    v["extra"] = v["beta"] * v["lambda_beta"]
    v["omega"] = v["theta"]
    v["s"] = 1 - v["omega"]
    assert v["omega"].lo >= 0 and v["omega"].hi <= 1
    v["h"] = v["beta"] + v["extra"] + v["eta_minus_model_2"]
    v["d"] = v["delta"] + v["mu"]
    v["gR"] = v["gamma"] / v["length_model_2"]
    v["gG"] = v["gamma"] / v["length_output"]
    v["kG"] = (v["gG"] + v["kappa_minus"]) / v["kappa_plus"] / (v["length_output"] / v["ribo_footprint_size"])
    v["kR"] = (v["gR"] + v["kappa_minus"]) / v["kappa_plus"] / (v["length_model_2"] / v["ribo_footprint_size"])
    v["R"] = v["total_rtl"]
    v["vg"] = v["gG"] * v["R"] / v["kG"]
    return v


def delivery_cover(cell):
    from decimal import Decimal, localcontext, ROUND_HALF_EVEN
    f = json.loads((ROOT / "committed_fit.json").read_text(), parse_float=Decimal, parse_int=Decimal)
    b = json.loads((ROOT / "model_config.json").read_text(), parse_float=Decimal, parse_int=Decimal)
    nu = f["parameters"]["nuisance_per_arm"][cell + "_model_2_RNA"]
    with localcontext() as ctx:
        ctx.prec = 75
        ctx.rounding = ROUND_HALF_EVEN
        ctx.Emin, ctx.Emax, ctx.clamp = -999999, 999999, 0
        m, kappa, power = b["initial_conditions_RNA"]["RNA_Burden"], nu["kappa_burden_circuit"], nu["p_dose_circuit"]
        assert m > 0 and kappa > 0 and power > 0
        low = (m * kappa).next_minus()
        z = (Decimal(5).ln().next_plus() * power).next_plus().exp().next_plus()
        high = ((m * kappa).next_plus() * z).next_plus()
    kl, ku = float(low), float(high)
    while Decimal.from_float(kl) > low:
        kl = float(DOWN(kl))
    while Decimal.from_float(ku) < high:
        ku = float(UP(ku))
    return kl, ku


def delivery_point_cover(cell, q):
    from decimal import Decimal, localcontext
    f = json.loads((ROOT / "committed_fit.json").read_text(), parse_float=Decimal, parse_int=Decimal)
    b = json.loads((ROOT / "model_config.json").read_text(), parse_float=Decimal, parse_int=Decimal)
    nu = f["parameters"]["nuisance_per_arm"][cell + "_model_2_RNA"]
    with localcontext() as ctx:
        ctx.prec = 80
        base = b["initial_conditions_RNA"]["RNA_Burden"]
        kap, power = nu["kappa_burden_circuit"], nu["p_dose_circuit"]
        ratio = Decimal(str(q))/20
        logratio = ratio.ln()
        lo = (((logratio.next_minus()*power).next_minus().exp().next_minus() * base).next_minus()*kap).next_minus()
        hi = (((logratio.next_plus()*power).next_plus().exp().next_plus() * base).next_plus()*kap).next_plus()
    kl, ku = float(lo), float(hi)
    while Decimal.from_float(kl) > lo: kl = float(DOWN(kl))
    while Decimal.from_float(ku) < hi: ku = float(UP(ku))
    return kl, ku


def time_grid(T1=96.):
    return np.unique(np.r_[np.linspace(0, .01, 101), np.linspace(.01, .1, 91),
                           np.linspace(.1, 1, 91), np.linspace(1, 24, 461),
                           np.linspace(24, 48, 241), np.linspace(48, T1, int((T1 - 48) / .2) + 1)])


def jets(x, ts, k0, eps, p):
    nc, nt = x.shape[:2]
    out = np.zeros((nc, nt, 3, 4))
    out[..., 0] = x
    E = np.exp(-p.beta * ts)[None, :]
    Ej = [E * (-p.beta) ** j / math.factorial(j) for j in range(4)]
    for n in range(3):
        den, force, production = [], [], []
        for j in range(n + 1):
            a = out[..., 0, j] - p.s * out[..., 1, j]
            den.append((1 if j == 0 else 0) + (p.r0 / p.kR + k0[:, None] / p.kG) * Ej[j] + eps[:, None] * a / p.kG)
            force.append(p.gR * p.R * p.r0 / p.kR * Ej[j])
            z = force[-1] + np.zeros_like(den[0])
            for k in range(1, j + 1):
                z -= den[k] * production[j - k]
            production.append(z / den[0])
        binding = -p.h * out[..., 1, n]
        for j in range(n + 1):
            binding += p.e * (out[..., 2, j] - out[..., 1, j]) * (out[..., 0, n - j] - out[..., 1, n - j])
        out[..., 0, n + 1] = (-p.beta * out[..., 0, n] - p.extra * out[..., 1, n]) / (n + 1)
        out[..., 1, n + 1] = binding / (n + 1)
        out[..., 2, n + 1] = (production[n] - p.d * (out[..., 2, n] - out[..., 1, n])) / (n + 1)
    return out


def matmul_up(a, b):
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    out = np.zeros(shape)
    for k in range(a.shape[-1]):
        out = UP(out + UP(a[..., :, k, None] * b[..., None, k, :]))
    return out


def affine_flow(A, f, tau):
    """Upper enclosure of a constant Metzler affine flow via positive series."""
    nc = len(A)
    B = np.zeros((nc, 4, 4))
    B[:, :3, :3], B[:, :3, 3] = A, f
    q = np.max(-np.diagonal(B, axis1=1, axis2=2), axis=1)
    N = B.copy()
    for i in range(4):
        N[:, i, i] = UP(N[:, i, i] + q)
    assert np.min(N) >= 0
    norm = I(np.sum(N, axis=2))
    # Enclose the exact positive row sums without trusting np.sum rounding.
    norm = I(np.zeros((nc, 4)))
    for i in range(4):
        norm = norm + I(N[:, :, i])
    maxnorm = np.max(norm.hi, axis=1)
    radius = (I(maxnorm) * tau).hi
    scaling = max(0, int(math.ceil(math.log2(max(float(radius.max()), 1e-300) / .5))))
    small = tau / (2 ** scaling)
    identity = np.broadcast_to(np.eye(4), B.shape).copy()
    term, total = identity.copy(), identity.copy()
    for k in range(1, 19):
        term = UP(matmul_up(term, N) * (small / k).hi)
        total = UP(total + term)
    z = I(maxnorm) * small
    tail = I(np.ones(nc))
    for k in range(1, 20):
        tail = tail * z / k
    tail = tail / (1 - z / 20)
    total = UP(total + tail.hi[:, None, None])
    total = UP(total * c.exp_neg(I(q) * small).hi[:, None, None])
    for _ in range(scaling):
        total = matmul_up(total, total)
    return total


def action(phi, x):
    z = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return matmul_up(phi[:, :3, :], z[:, :, None])[:, :, 0]


def dot_interval(A, w):
    out = I(np.zeros(A.shape[:-1]))
    for i in range(A.shape[-1]):
        out = out + I(A[..., i]) * I(w[..., i])
    return out


def cell_supremum(A, f, x, tau):
    # Numerical eigenvectors only propose weights; the row inequalities are
    # reevaluated with interval arithmetic and require no eigensolver accuracy.
    eig, vec = np.linalg.eig(A)
    indices = np.argmax(eig.real, axis=1)
    w = np.take_along_axis(vec, indices[:, None, None], axis=2)[:, :, 0].real
    w = np.maximum(np.abs(w), 1e-8)
    w /= np.max(w, axis=1)[:, None]
    assert np.all(np.isfinite(w)) and np.min(w) > 0
    rows = dot_interval(A, w[:, None, :]) / I(w)
    gamma = np.maximum(0, np.max(rows.hi, axis=1))
    normx = np.max((I(x) / I(w)).hi, axis=1)
    normf = np.max((I(f) / I(w)).hi, axis=1)
    growth = c.exp_pos(I(gamma) * tau)
    sup = (I(w) * (growth * (I(normx) + tau * I(normf)))[:, None]).hi
    return sup, float(np.max((I(gamma) * tau).hi))


def tail_resolvent(v, es, ks, env, terminal, t1):
    ne, nd = len(es) - 1, len(ks) - 1
    b1 = I(np.repeat(ks[1:], ne)) / v["kG"] * c.exp_neg(v["beta"] * I(t1))
    F1 = v["gR"] * v["R"] * v["r0"] / v["kR"] * c.exp_neg(v["beta"] * I(t1))
    M1 = v["M0"] * c.exp_neg(v["beta"] * I(t1))
    aa = I(float((v["beta"] + v["extra"]).lo));bb = I(float(v["extra"].hi))
    cc = I((v["eta_minus_model_2"] + v["e"] * M1).hi);dd = I(float((v["beta"] + v["eta_minus_model_2"]).lo));ee = I((v["e"] * M1).hi)
    ff = I((v["d"] + v["omega"] * F1 / v["kG"]).hi)
    gg = I((v["d"] + v["s"] * F1 / v["kG"]).hi);hh = I(float(v["d"].lo))
    den = dd - cc * bb / aa - ee / hh * (gg + ff * bb / aa)
    assert np.min(den.lo) > 0, ("tail matrix does not pass M-matrix test", float(den.lo.min()))
    r0, r1, r2 = [I(terminal[:, j]) for j in range(3)]
    r2 = r2 + F1 * b1 / (2 * v["beta"])
    P = (r1 + cc * r0 / aa + ee / hh * (r2 + ff * r0 / aa)) / den
    N = (r0 + bb * P) / aa
    return (v["omega"] * N + v["s"] * P).hi, float(den.lo.min())


def validate_requested_grid(ts, ks, es, nd, ne, T1):
    """Bind a cached proposal's grid to the requested tail start and shape."""
    if len(ts) == 0 or not np.isfinite(T1) or ts[-1] != T1:
        raise AssertionError("cached time cover must end at the requested analytic-tail start")
    if len(ks) != nd + 1 or len(es) != ne + 1:
        raise AssertionError("cached parameter grid shape differs from the requested grid")


def run(cell="reference", nd=16, ne=8, T1=96., point=None):
    started = time.time()
    import model_2_rectangle_tubes as tubes
    p, _, _ = load("model_2")
    tag = cell if point is None else f"{cell}_point{point:g}"
    cover = delivery_cover if point is None else lambda unused: delivery_point_cover(cell, point)
    if point is not None: nd = 1
    data = np.load(tubes.prepare(ROOT, tag, nd, ne, T1, cover, time_grid, jets, p))
    v = constants()
    ts, ks, es = [data[k] for k in ["ts", "ks", "es"]]
    validate_requested_grid(ts, ks, es, nd, ne, T1)
    kl, ku = cover(tag)
    assert ks[0] <= kl and ks[-1] >= ku
    env = tubes.enclose(data, v, flow=affine_flow, action=action)
    np.savez_compressed(ROOT/f"validated_env_{tag}_{nd}_{ne}_{T1:g}.npz",**env)
    print(json.dumps({"phase": "validated_rectangle_tubes", "cell": cell, "max_error": float(env["error"].max()),
                      "max_residual": float(env["residual"].max())}), flush=True)
    nc = nd * ne
    klo, khi = I(np.repeat(ks[:-1], ne)), I(np.repeat(ks[1:], ne))
    elo, ehi = I(np.tile(es[:-1], nd)), I(np.tile(es[1:], nd))
    x = np.zeros((nc, 3));Q = np.zeros(nc);S = np.zeros(nc)
    all_ok = np.ones(nc, dtype=bool)
    out = {24: np.zeros(nc), 48: np.zeros(nc)}
    bridge = np.zeros(nc)
    maximum_interior = -math.inf
    max_weighted_growth_step = 0.
    for it, t in enumerate(ts[:-1]):
        te = ts[it + 1];tau = I(te) - I(t)
        E = c.exp_neg(v["beta"] * I(t, te))
        B = 1 + v["r0"] / v["kR"] * E + I(klo.lo, khi.hi) / v["kG"] * E
        b = I(klo.lo, khi.hi) / v["kG"] * E
        F = v["gR"] * v["R"] * v["r0"] / v["kR"] * E
        ml, mu, zl, zu, al, au = [env[n][:, it] for n in ["ml", "mu", "zl", "zu", "al", "au"]]
        D = B + I(elo.lo, ehi.hi) * I(al, au) / v["kG"]
        U = I((ehi * F / (v["kG"] * I(D.lo) * I(D.lo))).hi)
        A = np.zeros((nc, 3, 3));force = np.zeros((nc, 3))
        A[:, 0, 0] = -float((v["beta"] + v["extra"]).lo);A[:, 0, 1] = float(v["extra"].hi)
        A[:, 1, 0] = (v["eta_minus_model_2"] + v["e"] * I(mu)).hi
        A[:, 1, 1] = -(v["e"] * I(zl) + v["beta"] + v["eta_minus_model_2"] + v["e"] * I(ml)).lo
        A[:, 1, 2] = (v["e"] * I(mu)).hi
        A[:, 2, 0] = (v["omega"] * I(np.zeros_like(U.hi), U.hi) - v["d"]).absmax();A[:, 2, 1] = (v["d"] + v["s"] * U).hi;A[:, 2, 2] = -float(v["d"].lo)
        force[:, 2] = (F * b / (I(D.lo) * I(D.lo))).hi
        sup, stepgrowth = cell_supremum(A, force, x, tau)
        max_weighted_growth_step = max(max_weighted_growth_step, stepgrowth)
        positive = v["omega"] * I(sup[:, 0]) + v["s"] * I(sup[:, 1])
        Vup = (v["vg"] * (I(B.hi) * positive / (I(D.lo) * I(D.lo)) - I(al) * I(b.lo) / (I(D.hi) * I(D.hi)))).hi
        if te <= 48:
            decay = c.exp_neg(v["d"] * tau);weight = (1 - decay) / v["d"]
            Qnext = (decay * I(Q) + weight * I(Vup)).hi
            if it == 0:
                first_forcing = Vup.copy();all_ok &= Vup < 0
            else:
                cap = (I(S) + tau * I(np.maximum(np.maximum(Q, Qnext), 0))).hi
                maximum_interior = max(maximum_interior, float(cap.max()));all_ok &= cap < 0
            S = (I(S) + weight * I(Q) + (tau - weight) / v["d"] * I(Vup)).hi
            Q = Qnext
            for T in out:
                if te <= T:
                    lam_start = (1 - c.exp_neg(v["d"] * (I(T) - I(t)))) / v["d"]
                    lam_end = (1 - c.exp_neg(v["d"] * (I(T) - I(te)))) / v["d"]
                    pos = I(lam_start.hi) * I(B.hi) * positive / (I(D.lo) * I(D.lo))
                    neg = I(np.maximum(0, lam_end.lo)) * I(al) * I(b.lo) / (I(D.hi) * I(D.hi))
                    out[T] = (I(out[T]) + tau * v["vg"] * (pos - neg)).hi
            if te == 48:
                S48, Q48 = np.minimum(S, out[48]), Q.copy()
        else:
            bridge = (I(bridge) + tau * positive / I(B.lo)).hi
        phi = affine_flow(A, force, tau)
        x = action(phi, x)
        assert np.all(np.isfinite(x)) and np.all(x >= 0)
        if (it + 1) % 500 == 0:
            print(json.dumps({"phase": "comparison", "time": float(te), "max_S": float(S.max()), "max_abs_sensitivity_bound": float(x.max())}), flush=True)
    tail, determinant_margin = tail_resolvent(v, es, ks, env, x, T1)
    later = (I(S48) + I(np.maximum(0, Q48)) / v["d"] + v["vg"] / v["d"] * (I(bridge) + I(tail))).hi
    result = {"mechanism": "model_2", "cell": cell, "modality": "RNA", "dose_ng": [20, 100] if point is None else [point, point], "loading": [0, 1],
              "dose_cells": nd, "loading_cells": ne, "time_cells": len(ts) - 1, "bridge_end_hours": T1,
              "max_state_tube_radius": float(env["error"].max()), "max_residual": float(env["residual"].max()),
              "upper_cumulative_log_burden_slope": {str(T): float(out[T].max()) for T in out},
              "all_positive_horizons_through48": bool(np.all(all_ok)),
              "max_first_cell_production_upper": float(first_forcing.max()),
              "max_interior_cap": maximum_interior,
              "upper_all_T_ge48": float(later.max()), "tail_M_matrix_margin": determinant_margin,
              "all_later_rectangles_strict": bool(np.all(later < 0)),
              "maximum_weighted_flow_growth_per_cell": max_weighted_growth_step,
              "seconds": time.time() - started,
              "method": "parameter-degree(3,2)-time-degree7 Bernstein residual tubes; free-RNA absolute-sensitivity Metzler flow; first-cell/interior tests; finite bridge and analytic M-matrix tail",
              "arithmetic": "custom outward-rounded IEEE754 binary64 primitives; explicit Decimal delivery cover"}
    result["passed"] = bool(np.all(all_ok) and np.all(later < 0) and all(float(z.max()) < 0 for z in out.values()))
    (ROOT / f"model_2_certificate_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(ROOT / f"model_2_certificate_cells_{tag}.npz", ks=ks, es=es,
                        upper24=out[24].reshape(nd, ne), upper48=out[48].reshape(nd, ne),
                        later=later.reshape(nd, ne), all_time_ok=all_ok.reshape(nd, ne),
                        terminal_absolute=x, bridge=bridge, tail=tail)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="reference", choices=["reference", "alternate"])
    parser.add_argument("--dose-cells", type=int, default=16)
    parser.add_argument("--loading-cells", type=int, default=8)
    parser.add_argument("--tail-start", type=float, default=96.)
    parser.add_argument("--point", type=float)
    args = parser.parse_args()
    run(args.cell, args.dose_cells, args.loading_cells, args.tail_start, args.point)
