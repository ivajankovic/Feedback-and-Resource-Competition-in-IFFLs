#!/usr/bin/env python3
"""Reproduce the fixed-model predictions and cumulative-ordering certificates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, json, math, os, platform, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
FIT = "119640ce00c1bfbb28f74964d5128a4a12a7f4fccdab3679ef9dd8646223b517"
CONFIG = "abfe9d9491c5c2fd4b91530d468ab28bd3182b9281600b9f7b4aa8e85661f6f0"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value): path.write_text(json.dumps(value, indent=2)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick",action="store_true",help="ODE, sensitivities, metrics and result figures.")
    mode.add_argument("--full",action="store_true",help="Quick results plus fresh interval certificates and cover audits.")
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=3,help="Parallel model_2 units in a full run.")
    parser.add_argument("--dry-run",action="store_true",help="Print commands without running them.")
    args=parser.parse_args()
    base="repro/model_2_continuous/full32x16"
    steps=[
      ("ODE and sensitivities",["checks/verify_repression_independent.py"],"checks/repression_independent_validation.json","independent",[]),
      ("Dose/loading grid",["checks/verify_repression_grid.py"],"checks/repression_grid_independent_validation.json","grid",[]),
      ("50 ng metrics",["checks/replay_50ng_thresholds.py","--package-root",".","--output","checks/replay_50ng_thresholds.json"],"checks/replay_50ng_thresholds.json","passed",[]),
      ("Result figures",["checks/replay_repression_figures.py","--package-root","."],"checks/fresh_repression_figure_replay.json","passed",[]),
    ]
    if args.full:
      steps += [
        ("model_1 certificate",["repro/replay_certificates.py","--fresh"],"repro/replay_report.json","model_1",["repro/certificate_reference.json","repro/tail_certificate_reference.json"]),
        ("model_2 continuous certificate",[base+"/replay_adaptive_certificate.py","--fresh","--workers",str(args.workers)],base+"/safe_adaptive_replay_report.json","model_2",[base+"/adaptive_paired_reports","repro/model_2_continuous/tensor_audit/adaptive_cover_audit.json"]),
        ("Paired tail margins",[base+"/capture_adaptive_pairs.py"],base+"/adaptive_paired_reports/paired_summary.json","passed",[]),
        ("Independent cover audit",["repro/model_2_continuous/tensor_audit/audit_adaptive_cover.py"],"repro/model_2_continuous/tensor_audit/adaptive_cover_audit.json","passed",[]),
        ("Certificate record audit",["checks/audit_model_2_archived_records.py","--root",".","--output","checks/audit_model_2_archived_records.json"],"checks/audit_model_2_archived_records.json","passed",[]),
      ]
    if args.dry_run:
      print(json.dumps([[sys.executable,*s[1]] for s in steps],indent=2));return 0
    assert sys.float_info.mant_dig==53 and sys.float_info.rounds==1
    assert sha(ROOT/"repro/committed_fit.json")==FIT and sha(ROOT/"repro/model_config.json")==CONFIG
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+f"_{os.getpid()}"
    folder=ROOT/"checks/runs"/run_id;folder.mkdir(parents=True)
    lock=ROOT/"checks/.reproduction_lock"
    try:lock.mkdir()
    except FileExistsError:raise SystemExit("Another run is active; do not overlap runs in one folder.")
    report={"run_id":run_id,"started_at_utc":now(),"mode":"full" if args.full else "quick","completed":False,"passed":False,"steps":[],"scope":"Fixed RNA model predictions and certificates; no experimental data or fitting.","input_sha256":{"repro/committed_fit.json":FIT,"repro/model_config.json":CONFIG},"runner_sha256":sha(Path(__file__)),"environment":{"python":sys.version,"platform":platform.platform(),"packages":{k:importlib.metadata.version(k) for k in ["numpy","scipy","matplotlib","pandas"]}}}
    latest=ROOT/"checks/latest_run.json"
    def checkpoint():write(folder/"report.json",report);write(latest,report)
    started=time.perf_counter();checkpoint()
    env=os.environ|{"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","MPLBACKEND":"Agg"}
    try:
      for index,(name,arguments,output,gate,invalidate) in enumerate(steps,1):
        for rel in [output,*invalidate]:
          path=ROOT/rel
          if path.is_dir():shutil.rmtree(path)
          else:path.unlink(missing_ok=True)
        step={"name":name,"started_at_utc":now(),"passed":False,"command":[sys.executable,*arguments]};report["steps"].append(step);checkpoint()
        log=folder/f"{index:02d}.log";before=time.perf_counter();print(f"{index}/{len(steps)} {name}",flush=True)
        with log.open("w") as stream:
          result=subprocess.run([sys.executable,*arguments],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        step.update(returncode=result.returncode,seconds=time.perf_counter()-before,log=str(log.relative_to(ROOT)))
        assert result.returncode==0,f"{name} failed: {log}"
        data=json.loads((ROOT/output).read_text())
        if gate=="independent":
          assert len(data["rows"])==6 and len(data["checks"])==12
          assert math.isfinite(data["max_solver_scaled_error"]) and data["max_solver_scaled_error"]<1e-7
          assert math.isfinite(data["max_fd_scaled_error"]) and data["max_fd_scaled_error"]<1e-6
        elif gate=="grid":
          assert len(data["rows"])==54 and data["max_archive_error"]<1e-7 and data["max_sampled_S_theta"]<0
        else:
          assert data.get("passed") is True,f"{name} did not pass its gate"
          if gate=="model_1":assert data["proposal_source"]=="regenerated" and all(data["gates"].values())
          if gate=="model_2":assert data["complete_replay"] and data["fresh_ordinary_proposals"] and data["full_domain_certified"] and len(data["blocks"])==9
        assert sha(ROOT/"repro/committed_fit.json")==FIT and sha(ROOT/"repro/model_config.json")==CONFIG
        step.update(passed=True,completed_at_utc=now(),report=output,report_sha256=sha(ROOT/output));checkpoint()
        print(f"  passed ({step['seconds']:.1f} s)",flush=True)
      if args.full:
        subprocess.run([sys.executable,"export_grids.py"],cwd=ROOT,env=env,check=True)
        report["fresh_full_interval_certificates_passed"]=True
      report.update(passed=True,completed=True);return 0
    except (Exception,KeyboardInterrupt) as error:
      report["error"]=f"{type(error).__name__}: {error}";print(report["error"],file=sys.stderr);return 1
    finally:
      report.update(elapsed_seconds=time.perf_counter()-started,finished_at_utc=now());checkpoint();shutil.rmtree(lock)
      print("Results: "+str(latest),flush=True)

if __name__=="__main__":raise SystemExit(main())
