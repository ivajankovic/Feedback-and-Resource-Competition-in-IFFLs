"""Regenerate Figures 4/5 from fresh fixed-fit ODE and grid calculations.

This is ordinary numerical reproduction, not an interval certificate or an
experimental validation. Run verify_repression_grid.py before this script.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.package_root.resolve()
    started = time.perf_counter()
    solver = root / "checks/verify_repression_independent.py"
    spec = importlib.util.spec_from_file_location("independent_repression_figures", solver)
    repression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repression)
    figures = root / "figures"
    old = np.load(figures / "response_plot_data.npz")
    old_arrays = {key: old[key].copy() for key in old.files}
    old.close()
    arrays = {}
    peaks = {"description": "Fresh q=20 ng reference peaks obtained by scalar maximization; same fixed fit and model output units as the manuscript."}
    checks = []
    for mechanism in ("model_1", "model_2"):
        times = np.linspace(0.0, 48.0, 1601)
        arrays[mechanism + "_time"] = times
        for loading, label in ((0.0, "omitted"), (1.0, "included")):
            sol, constants = repression.solve(mechanism, 20.0, loading,
                                       method="Radau", rtol=2e-12)
            values = sol.sol(times)
            arrays[mechanism + "_" + label] = values
            prior = old_arrays[mechanism + "_" + label]
            error = float(np.max(np.abs(values - prior) /
                                 np.maximum(1.0, np.abs(values))))
            checks.append({"mechanism": mechanism, "loading": loading,
                           "max_scaled_difference_from_archived_curves": error,
                           "archive_agreement_tolerance": 1e-7,
                           "passed": error < 1e-7})
            if loading == 0.0:
                peaks[mechanism], peak_time = repression.maximum(lambda t: sol.sol(t)[3])
                peaks[mechanism] = float(peaks[mechanism])
                peaks[mechanism + "_reference_peak_time_h"] = float(peak_time)
    if not all(check["passed"] for check in checks):
        raise AssertionError(checks)
    np.savez_compressed(figures / "response_plot_data.npz", **arrays)
    (figures / "response_peak_normalization.json").write_text(
        json.dumps(peaks, indent=2) + "\n")
    grid = root / "checks/repression_grid_independent_validation.json"
    grid_data = json.loads(grid.read_text())
    assert len(grid_data["rows"]) == 54
    shutil.copy2(grid, figures / grid.name)
    command = [sys.executable, str(figures / "make_repression_figures.py"),
               "--data-dir", str(figures), "--output-dir", str(figures)]
    subprocess.run(command, cwd=root, check=True)
    inputs = [solver, root / "repro/committed_fit.json",
              root / "repro/model_config.json", grid,
              figures / "make_repression_figures.py"]
    report = {
        "passed": True,
        "scope": "Fresh ordinary fixed-fit predictions; not interval certification, experimental validation, or uncertainty propagation.",
        "dose_ng": 20,
        "trajectory_horizon_h": 48,
        "trajectory_time_points": 1601,
        "solver": {"method": "Radau", "rtol": 2e-12, "atol": 2e-14},
        "grid_rows": len(grid_data["rows"]),
        "curve_checks": checks,
        "source_sha256": {str(path.relative_to(root)): sha(path) for path in inputs},
        "output_sha256": {str(path.relative_to(root)): sha(path) for path in
                           [figures / "response_plot_data.npz",
                            figures / "response_peak_normalization.json",
                            figures / "repression_response_metrics.pdf",
                            figures / "repression_loading_effects.pdf"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (root / "checks/fresh_repression_figure_replay.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
