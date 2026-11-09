#!/usr/bin/env python3
"""Replay the reported 50 ng/full-loading expression discrepancies.

From any working directory, run the delivered package's script:
    python3 /path/to/package/checks/replay_50ng_thresholds.py

Or point this file at a package explicitly:
    python3 replay_50ng_thresholds.py --package-root /path/to/package

This imports the retained independent free-species implementation in
checks/verify_repression_independent.py. It does not copy the reduced equations,
refit parameters, rebuild expression data or replay the interval certificates.
Requires NumPy and SciPy. Writes a JSON report and exits nonzero on mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy


RTOL = 2e-12
ATOL = RTOL / 100  # The retained free-species solve() uses this exact rule.
ABS_TOL_PERCENTAGE_POINTS = 1e-6
EXPECTED = {
    "model_1": {
        "abs_E48_percent": 6.743700356332216,
        "Dinf48_percent": 11.283144328475153,
    },
    "model_2": {
        "abs_E48_percent": 6.3449638763849014,
        "Dinf48_percent": 10.019998150445776,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Extracted package containing checks/ and repro/; defaults to the script's package root.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="JSON destination; defaults to PACKAGE_ROOT/checks/replay_50ng_thresholds.json.",
    )
    args = parser.parse_args()
    root = args.package_root.resolve()
    output = (args.output or root / "checks/replay_50ng_thresholds.json").resolve()
    independent_path = root / "checks/verify_repression_independent.py"
    input_paths = [
        independent_path,
        root / "repro/committed_fit.json",
        root / "repro/model_config.json",
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        parser.error("Missing retained package input(s): " + ", ".join(missing))

    spec = importlib.util.spec_from_file_location("retained_free_species_replay", independent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {independent_path}")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)

    start = time.perf_counter()
    rows = []
    checks = []
    for mechanism in ("model_1", "model_2"):
        methods = {}
        for method in ("Radau", "BDF"):
            reference, constants = independent.solve(
                mechanism, 50.0, 0.0, method=method, rtol=RTOL, T=48.0,
            )
            loaded, _ = independent.solve(
                mechanism, 50.0, 1.0, method=method, rtol=RTOL, T=48.0,
            )
            metrics = independent.metrics(reference, loaded)
            values = {
                "abs_E48_percent": float(100 * abs(metrics["E"])),
                "Dinf48_percent": float(100 * metrics["Dinf"]),
                "reference_C48": float(reference.sol(48.0)[4]),
                "loaded_C48": float(loaded.sol(48.0)[4]),
                "k0_C": float(constants["k0"]),
                "p_del": float(constants["pdel"]),
            }
            methods[method] = values
            for quantity, expected in EXPECTED[mechanism].items():
                discrepancy = abs(values[quantity] - expected)
                checks.append({
                    "mechanism": mechanism, "method": method,
                    "check": "reported_quantity", "quantity": quantity,
                    "expected_percent": expected,
                    "observed_percent": values[quantity],
                    "absolute_difference_percentage_points": discrepancy,
                    "passed": bool(np.isfinite(discrepancy) and discrepancy <= ABS_TOL_PERCENTAGE_POINTS),
                })
        for quantity in EXPECTED[mechanism]:
            discrepancy = abs(methods["Radau"][quantity] - methods["BDF"][quantity])
            checks.append({
                "mechanism": mechanism, "check": "Radau_BDF_crosscheck",
                "quantity": quantity,
                "absolute_difference_percentage_points": discrepancy,
                "passed": bool(np.isfinite(discrepancy) and discrepancy <= ABS_TOL_PERCENTAGE_POINTS),
            })
        waveform_only = all(
            values["abs_E48_percent"] < 10.0 < values["Dinf48_percent"]
            for values in methods.values()
        )
        checks.append({
            "mechanism": mechanism,
            "check": "only_waveform_exceeds_illustrative_10_percent",
            "passed": waveform_only,
        })
        rows.append({"mechanism": mechanism, "methods": methods})

    record = {
        "scope": "Ordinary numerical replay through the retained independent free-species ODE implementation at one committed fit. Radau/BDF agreement is a numerical diagnostic, not a validated interval error bound or experimental time-course validation.",
        "nominal_burden_dose_ng": 50.0,
        "reference_loading": 0.0,
        "compared_loading": 1.0,
        "observation_horizon_h": 48.0,
        "solvers": {"methods": ["Radau", "BDF"], "rtol": RTOL, "atol": ATOL, "dense_output": True},
        "normalization": {
            "abs_E48_percent": "100*abs(C48_loaded/C48_reference-1); C48 is the integrated unscaled Output concentration.",
            "Dinf48_percent": "100*max_[0,48]|Y_loaded-Y_reference|/max_[0,48]Y_reference; both use the same mechanism and dose.",
            "waveform_extrema": "Retained maximum(): endpoints plus local maxima bracketed on 3001 uniform times and refined by bounded scalar optimization with xatol=2e-11 h.",
            "model_concentration_unit": "C; no physical concentration/fluorescence calibration is inferred.",
        },
        "acceptance_absolute_tolerance_percentage_points": ABS_TOL_PERCENTAGE_POINTS,
        "tolerance_scope": "Numerical replay agreement with four reported quantities and between solvers; separate from the illustrative biological/modeling tolerance of 10%.",
        "results": rows,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "runtime_seconds": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "input_sha256": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
