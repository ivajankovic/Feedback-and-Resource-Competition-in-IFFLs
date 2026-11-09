# Feedback and Resource Competition in Incoherent Feedforward Loops: When Is Cumulative Dose Ordering Preserved?

Reproducibility check for the paper's fixed-model calculations. The scripts regenerate the response figures, expression metrics and cumulative dose-ordering certificates from the supplied model parameters.

Model I blocks translation of bound target RNA (`model_1` in code); Model II accelerates bound-RNA decay while allowing residual translation (`model_2`). Both models share translation resources between output and repressor production. Each is compared with target demand omitted or included.

The parameterizations come from a joint fit to unpublished expression measurements. No experimental measurements, data summaries or fitting diagnostics are distributed here, so this archive reproduces the fixed-model calculations rather than the original fit.

## Run

Tested with Python 3.12. Install the pinned dependencies, then run:

```sh
python -m pip install -r requirements.txt
python reproduce.py --quick
```

For fresh interval validation of both models:

```sh
python reproduce.py --full --workers 3
```

`--quick` checks model trajectories, sensitivities and metrics and regenerates Figures 4 and 5. It is not an interval proof. `--full` additionally rebuilds the validated state enclosures, checks every accepted parameter rectangle and positive finite horizon, and verifies the paired tail budgets. It regenerates omitted temporary caches. Do not run two checks simultaneously in the same extracted folder.

The current execution report is `checks/latest_run.json`; individual logs are under `checks/runs/`. Failed or missing fresh outputs stop the check.

## Files

- `parameters/`: constants with units and fitted/fixed/shared status, delivery maps and initial conditions.
- `repro/`: model equations, parameter-only inputs and interval certificate code and records.
- `checks/`: independent ODE, sensitivity and certificate checks.
- `figures/`: the TikZ schematic, plotting code, simulated curves and generated figure files.
- `results/`: reference numerical predictions used for comparison; these are not measurements.
- `certificate_rectangles.csv`, `certificate_time_nodes.csv`: accepted parameter cells and time grids.
- `NUMERICS.md`: implementation, arithmetic and horizon details.

Time is in hours and concentration in model units `C`, without a physical concentration calibration. Target and regulator delivery are fixed; only burden varies. The reported guarantees concern this fixed parameterization, not parameter uncertainty or experimental validation of predicted time courses.

The trajectory plots divide both compared responses by the same reference peak at zero target loading. The metrics use `E_T=(C_T-C_T^0)/C_T^0`, `R_T=1-S_T/S_T^0` and `D_T^infinity=max_t|Y-Y^0|/max_t Y^0`; superscript0 denotes the reference at zero target loading and the same dose. Cumulative output has units `C h`.
