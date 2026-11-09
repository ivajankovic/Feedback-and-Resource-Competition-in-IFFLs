"""Safe replay entry point for the adaptive continuous model_2 certificate.

The original final coarse dose row is excluded; its two refined rows replace
it. Block07 retains only its first three rows. Unit8 is refine07last. A full
replay must pass the independent exact-cover audit (528 accepted rectangles).

Prior reports are archived before execution. A nonzero process exit, missing
fresh report, failed gate or changed executed source rejects the replay.
Ordinary proposal caches are untrusted and always residual-validated; --fresh
also regenerates them. Exact executed block sources remain unchanged.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
CORE=['certify.py','certify_model_2.py','model_2_rectangle_tubes.py','repression_mechanisms.py','committed_fit.json','model_config.json','run_block.py']
FIT='119640ce00c1bfbb28f74964d5128a4a12a7f4fccdab3679ef9dd8646223b517'
CONFIG='abfe9d9491c5c2fd4b91530d468ab28bd3182b9281600b9f7b4aa8e85661f6f0'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--blocks',nargs='+',type=int,default=list(range(9)));p.add_argument('--workers',type=int,default=3);p.add_argument('--fresh',action='store_true');a=p.parse_args()
    if len(set(a.blocks))!=len(a.blocks) or any(i not in range(9) for i in a.blocks):raise ValueError('Select distinct block indices0..8 (8 is the final-dose-cell refinement).')
    if a.workers not in [1,2,3]:raise ValueError('Use1..3workers to bound memory usage.')
    started=time.time();runid=time.strftime('%Y%m%dT%H%M%S')+f'_{os.getpid()}'
    archive=ROOT/'replay_records'/runid;archive.mkdir(parents=True)
    def work(i):
        unit=f'block{i:02d}' if i<8 else 'refine07last';folder=ROOT/unit;dest=archive/unit;dest.mkdir()
        source={n:sha(folder/n) for n in CORE}
        if source['committed_fit.json']!=FIT or source['model_config.json']!=CONFIG:raise AssertionError('Unexpected fit/config snapshot.')
        prior=folder/'block_result.json'
        if prior.exists():prior.replace(dest/'prior_block_result.json')
        if a.fresh:
            for cache in folder.glob('*proposals*.npz'):cache.unlink()
        before=time.time()
        with (dest/'run.log').open('w') as stream:
            process=subprocess.run([sys.executable,'run_block.py'],cwd=folder,
                env=os.environ|{'OPENBLAS_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT)
        status={'block':i,'process_returncode':process.returncode,'seconds':time.time()-before,'passed':False}
        if process.returncode!=0:
            status['error']='Certificate process did not exit successfully.'
        elif not prior.exists():
            status['error']='Fresh block_result.json was not produced.'
        else:
            result=json.loads(prior.read_text())
            unchanged=all(sha(folder/n)==h for n,h in source.items())
            hashes_match=result.get('source_sha256')==source
            cells=np.load(folder/'model_2_certificate_cells_reference.npz')
            selected_later=cells['later'][:3] if i==7 else cells['later']
            gates={
                'fresh_accepted_subcover_strict':bool(np.all(selected_later<0)) and (result.get('passed') is True or i==7),
                'source_unchanged':unchanged,'reported_hashes_match_executed_sources':hashes_match,
                'first_cell_strict':result.get('max_first_cell_production_upper',0)<0,
                'every_interior_horizon_strict':result.get('all_positive_horizons_through48') is True and result.get('max_interior_cap',0)<0,
                'all_later_accepted_horizons_strict':bool(np.all(selected_later<0)),
                'tail_stability':result.get('tail_M_matrix_margin',0)>0,
                'finite24_and48':set(result.get('upper_cumulative_log_burden_slope',{}))=={'24','48'} and all(v<0 for v in result.get('upper_cumulative_log_burden_slope',{}).values()),
                'requested_dimensions':result.get('dose_cells')==(2 if i==8 else 4) and result.get('loading_cells')==16 and result.get('bridge_end_hours')==96,
            }
            status['gates']=gates;status['passed']=all(gates.values());status['result_file']=str(prior.relative_to(ROOT))
            shutil.copyfile(prior,dest/'fresh_block_result.json')
        (dest/'replay_status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        return status
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(work,a.blocks))
    report={'passed':all(r['passed'] for r in rows),'complete_replay':set(a.blocks)==set(range(9)),'full_domain_certified':False,'requested_blocks':a.blocks,'fresh_ordinary_proposals':a.fresh,'workers':a.workers,'run_id':runid,'blocks':rows,'seconds':time.time()-started,
        'scope':'A fresh report and successful process exit are mandatory; final full-domain promotion additionally requires the independent cover audit.'}
    (archive/'safe_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'safe_adaptive_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['passed'] and set(a.blocks)==set(range(9)):
        audit=subprocess.run([sys.executable,str(ROOT.parent/'tensor_audit'/'audit_adaptive_cover.py')],capture_output=True,text=True)
        report['independent_cover_audit_returncode']=audit.returncode
        report['independent_cover_audit_stdout']=audit.stdout
        report['independent_cover_audit_stderr']=audit.stderr
        report['passed']=audit.returncode==0
        report['full_domain_certified']=report['passed']
        (ROOT/'safe_adaptive_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
        (archive/'safe_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
