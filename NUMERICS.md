# Numerical implementation

The calculations use the derivative `S_T = partial_(log k0) C_T`, where `k0` is delivered burden and `C_T` is the time integral of output concentration. Nominal-dose sensitivity is `(p_del/q)*S_T`; the conversion is positive. Parameter-only JSON inputs preserve the exact computational values. CSV tables state their units, status, delivery laws and initial conditions. Within each mechanism, the two loading configurations retain binding and differ only in target translation demand. The JSON field `theta` is the Model II bound-RNA translation fraction `omega`; it is distinct from the log-burden differentiation coordinate.

## Parameter and time grids

`certificate_rectangles.csv` lists all 784 accepted rectangles: 256 for Model I and 528 for Model II. Delivered-burden endpoints, rather than approximate inverse-map nominal labels, define the cover. The delivery endpoints are independently bracketed with 200-digit Decimal arithmetic. Every rectangle uses `T0=48 h` and `T1=96 h`.

| Model | Parameter cover | Time grid through 96 h |
|---|---|---|
| Model I | 32 geometric burden cells × 8 loading cells | 100 cells on 0–0.2 h, 2390 on 0.2–48 h, then 480 steps of 0.1 h |
| Model II | 33 burden cells × 16 loading cells | Steps 0.0001 h on 0–0.01; 0.001 on 0.01–0.1; 0.01 on 0.1–1; 0.05 on 1–24; 0.1 on 24–48; 0.2 on 48–96 |

`certificate_time_nodes.csv` retains every node. Model II starts from 32 burden cells; the entire last coarse cell is replaced by two narrower cells. Blocks 00–06 retain all rows, block07 retains rows 0–2, and `refine07last` provides the two replacement rows. Failed coarse cells are excluded. `python export_grids.py` regenerates the CSVs after a full run.

## State enclosures

Ordinary integration proposes polynomials; residual validation establishes the enclosures. Model I uses Radau corner solves with relative and absolute tolerances `2e-12`, degree-seven Hermite time polynomials and Bernstein residual ranges. The parameter order encloses intermediate states between opposite corners enlarged by validated errors. Its corner error bound is `48*rho*exp(48*G)`, where `rho` bounds the residual and `G` the logarithmic norm.

Model II uses LSODA proposals (`rtol=2e-11`, `atol=2e-12`) and Bernstein tensor degree `(3,2,7)` in burden, loading and time. All residuals are bounded over complete parameter–time boxes. Rational residuals are bounded after clearing the positive denominator. Initial errors, polynomial joins and propagated endpoint errors are retained. A candidate radius `R=exp(gamma*h)*v` is accepted only when outward-rounded inequalities verify `v>=e_j` and `gamma*v>=A_E(R)*v+rho`; failure after 30 enlargements aborts. Endpoint errors propagate to the next cell without resetting. Sensitivities are bounded from the variational equations, not derivatives of the interpolants.

Ranges for total RNA `M`, bound RNA `c` and total repressor `W` give `m=M-c`, `Z=W-c` and `a=M-(1-omega)*c`, intersected with physical nonnegativity and analytic RNA bounds. Full-cell interval evaluation of `exp(-beta*t)`, `b`, `B`, `F` and `D` then gives `m_L,m_U,Z_L,a_L,B_U,b_L,b_U,F_U,D_L,D_U`. In particular, `D_L=B_L+epsilon_L*a_L/kappa_G` and `D_U=B_U+epsilon_U*a_U/kappa_G`; the code verifies `D_L>0`.

## Arithmetic and cumulative bounds

The custom interval implementation assumes IEEE754 binary64 with 53 significant bits. Elementary operations expand to adjacent exterior numbers with `numpy.nextafter`; division rejects intervals containing zero. Exact time differences are enclosed before differentiation, integration and kernel evaluation. Exponentials use positive Taylor series with geometric remainders after scaling by 16, followed by four interval squarings; negative exponentials use reciprocals.

Sensitivity flows use uniformization and positive Taylor bounds: 40 terms plus a remainder for the two-state Model I comparison, and 18 terms plus a remainder and outward squaring for the three-state Model II comparison embedded in homogeneous coordinates. Candidate eigenvector weights require interval row-inequality checks. Finite-cell comparisons need not be Hurwitz.

Signed rectangular quadrature multiplies the positive production bound by the upper kernel at each cell's left endpoint and the negative contribution by its lower kernel at the right endpoint. The kernel decreases in time. Products and sums are outward rounded and control the entire cell.

## Every positive horizon and the tail

Initially target-RNA sensitivities vanish and `partial_theta V(0)=-v_G*M0*b(0)/D(0)^2<0`. Validated production bounds cover the first 0.002 h for Model I and first 0.0001 h for Model II. Exact affine propagation and an interior-cell supremum test cover all subsequent horizons through 48 h. No fixed negative constant is claimed as the horizon tends to zero.

Finite comparison covers 48–96 h. A constant Hurwitz Metzler matrix then bounds the remaining decaying forcing and its integral. Stability uses uniform M-matrix tests, not numerical eigenvalues. The Model I minimum two-state determinant exceeds 0.05405484; the Model II minimum Schur margin exceeds 0.09241540965.

Each rectangle retains its own negative accumulated margin and adverse tail budget. The limiting pairs are approximately `-47.74899438564626 +39.2072550754751` for Model I and `-1.9163650605263902 +0.2352685681293542` for Model II, giving later-horizon bounds `-8.541739310171158` and `-1.6810964923970355`. A global bound must not be combined with an unrelated rectangle's budget. The NPZ and paired reports retain the actual arrays and arithmetic.

`checks/latest_run.json` records fresh process exits, acceptance gates, hashes, environment and runtimes. The full command regenerates proposals, validates residuals and state tubes, propagates sensitivities, captures paired margins and independently audits coverage and records. Software/hardware is not formally verified. These calculations neither refit parameters nor validate biological time courses.
