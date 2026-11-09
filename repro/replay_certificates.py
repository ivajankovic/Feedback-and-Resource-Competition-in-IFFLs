"""Replay the committed-fit certificates, with explicit machine-readable gates."""
import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import scipy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIT_SHA="119640ce00c1bfbb28f74964d5128a4a12a7f4fccdab3679ef9dd8646223b517"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run(fresh=False):
    started=time.time()
    assert sys.float_info.mant_dig==53 and sys.float_info.rounds==1
    assert sha(HERE/"committed_fit.json")==FIT_SHA
    inputs=["committed_fit.json","model_config.json","model_equations.py",
            "source/models/Repressor_iFFL.json","source/simulator/model.py"]
    input_hashes={n:sha(HERE/n) for n in inputs}
    cache=HERE/"corner_polynomials_reference.npz"
    had_cache=cache.exists()
    if fresh and had_cache: cache.unlink()
    records=[]
    logs=HERE/"replay_logs"; logs.mkdir(exist_ok=True)
    for name,args in [("certify.py",["reference"]),("certify_tail.py",[])]:
        before=time.time()
        result=subprocess.run([sys.executable,str(HERE/name),*args],
            cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (logs/(Path(name).stem+".log")).write_text(result.stdout)
        record={"script":name,"sha256":sha(HERE/name),
                "exit_code":result.returncode,"seconds":time.time()-before}
        records.append(record)
        print(json.dumps(record),flush=True)
        if result.returncode:
            raise RuntimeError("Certificate replay failed; inspect replay_logs/"+Path(name).stem+".log")
    f=json.loads((HERE/"certificate_reference.json").read_text())
    t=json.loads((HERE/"tail_certificate_reference.json").read_text())
    gates={
        "complete_dose_loading_cover":f["dose_cells"]==32 and f["coupling_cells"]==8
            and f["dose_ng"]==[20,100] and f["coupling_interval"]==[0,1],
        "all_early_positive_horizons":f["all_positive_horizons_through_48"],
        "first_cell_strict":f["max_first_cell_production_slope_upper"] < -1.27,
        "later_early_cells_strict":f["max_finite_cell_interior_slope_cap"] < -2.54e-6,
        "published_24h_bound":f["upper_cumulative_log_burden_slope"]["24"] < -57.70,
        "published_48h_bound":f["upper_cumulative_log_burden_slope"]["48"] < -47.74,
        "every_tail_rectangle_strict":t["all_cells_strictly_negative"],
        "published_all_later_bound":t["max_all_later_horizon_upper"] < -8.54,
        "tail_comparison_hurwitz":t["minimum_tail_matrix_determinant"]>0,
        "fixed_inputs_unchanged":all(sha(HERE/n)==v for n,v in input_hashes.items()),
    }
    terminal=np.load(HERE/"certified_terminal_reference.npz")
    tail=np.load(HERE/"tail_certificate_cells_reference.npz")
    # The addition must be done per rectangle before maximizing.
    paired=terminal["Supper"].reshape(tail["extra"].shape)+tail["extra"]
    gates["paired_tail_cells"]=bool(np.all(tail["upper"]>=paired)
        and float(tail["upper"].max())==t["max_all_later_horizon_upper"])
    report={"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "passed":bool(all(gates.values())),"gates":gates,
        "scope":{"dose_ng":[20,100],"loading":[0,1],"horizons":"every finite T>0",
                 "fit":"reference_model_1_RNA committed fit; no refitting or parameter ensemble"},
        "proposal_source":"regenerated" if fresh or not had_cache else "reused and revalidated",
        "proposals_sha256":sha(cache),"inputs_sha256":input_hashes,"runs":records,
        "runtime":{"python":sys.version,"platform":platform.platform(),
                   "numpy":np.__version__,"scipy":scipy.__version__,
                   "binary64_mantissa_bits":sys.float_info.mant_dig,
                   "rounding_mode":sys.float_info.rounds},
        "reported_bounds":{"24h":f["upper_cumulative_log_burden_slope"]["24"],
            "48h":f["upper_cumulative_log_burden_slope"]["48"],
            "all_T_ge_48h":t["max_all_later_horizon_upper"]},
        "seconds":time.time()-started,
        "limits":["Custom interval implementation under stated IEEE754 assumptions.",
                  "Independent certification software audit remains outstanding.",
                  "Ordinary diagnostic solves and biological validation are separate."]}
    (HERE/"replay_report.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]: raise AssertionError(gates)
    print(json.dumps({"passed":True,"bounds":report["reported_bounds"]}),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh",action="store_true",help="regenerate all corner proposals")
    run(parser.parse_args().fresh)
