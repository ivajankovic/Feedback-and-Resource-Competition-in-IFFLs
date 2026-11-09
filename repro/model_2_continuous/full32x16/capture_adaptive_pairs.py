"""Comparison-only reporting replay of already validated state enclosures.

Executed certificate sources/outputs are left unchanged. This is not a fresh
state-tube proof; the original proof outputs must match bitwise.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent

def capture(index):
    import inspect,numpy as np
    unit=f'block{index:02d}' if index<8 else 'refine07last';block=ROOT/unit;out=ROOT/'adaptive_paired_reports'/unit;out.mkdir(parents=True,exist_ok=True)
    record=json.loads((block/'block_result.json').read_text());assert record['passed'] or index==7
    nd=2 if index==8 else 4
    for name,expected in record['source_sha256'].items():assert hashlib.sha256((block/name).read_bytes()).hexdigest()==expected,name
    sys.path.insert(0,str(block));import certify_model_2 as cert;import model_2_rectangle_tubes as tubes
    envpath=block/f'validated_env_reference_{nd}_16_96.npz'
    with np.load(envpath) as z:env={k:z[k] for k in z.files}
    interval=tuple(record['actual_delivered_interval']);cert.delivery_cover=lambda _:interval
    old_enclose=tubes.enclose;tubes.enclose=lambda *args,**kwargs:env
    save=np.savez_compressed;write=Path.write_text
    def save_report(path,*args,**payload):
        path=Path(path)
        if path.name.startswith('validated_env_'):
            assert set(payload)==set(env) and all(np.array_equal(payload[k],env[k]) for k in env)
            return
        if path.name.startswith('model_2_certificate_cells_'):
            loc=inspect.currentframe().f_back.f_locals
            payload=dict(payload)
            payload['S48']=np.asarray(loc['S48']).reshape(payload['upper48'].shape)
            payload['Q48']=np.asarray(loc['Q48']).reshape(payload['upper48'].shape)
            path=out/path.name
        save(path,*args,**payload)
    def write_report(path,data,*args,**kwargs):
        if path.parent==block and path.name.startswith('model_2_certificate_'):path=out/path.name
        return write(path,data,*args,**kwargs)
    np.savez_compressed=save_report;Path.write_text=write_report
    started=time.time()
    try:result=cert.run('reference',nd,16,96.)
    finally:np.savez_compressed=save;Path.write_text=write;tubes.enclose=old_enclose
    assert result['passed'] or index==7
    old=np.load(block/'model_2_certificate_cells_reference.npz');new=np.load(out/'model_2_certificate_cells_reference.npz')
    match={k:bool(np.array_equal(old[k],new[k])) for k in old.files};assert all(match.values()),match
    accepted_rows=3 if index==7 else nd
    assert np.all(new['later'][:accepted_rows]<0)
    ind=np.unravel_index(np.argmax(new['later'][:accepted_rows]),new['later'][:accepted_rows].shape)
    flat=int(np.ravel_multi_index(ind,new['later'].shape))
    v=cert.constants();I=cert.I
    E=I(max(0,float(new['Q48'][ind])))/v['d']+v['vg']/v['d']*(I(new['bridge'][flat])+I(new['tail'][flat]))
    pair=(I(new['S48'][ind])+E).hi
    report={'unit':unit,'accepted_original_dose_rows':list(range(accepted_rows)),'block_index':index,'scope':'Comparison-only reporting replay; original executed proof sources and interval outputs unchanged.',
        'source_hashes_verified':True,'all_original_arrays_bitwise_identical':all(match.values()),'array_checks':match,
        'validated_env_sha256':hashlib.sha256(envpath.read_bytes()).hexdigest(),'seconds':time.time()-started,
        'limiting_later_cell_index':list(map(int,ind)),'delivered_burden_interval':[float(new['ks'][ind[0]]),float(new['ks'][ind[0]+1])],
        'loading_interval':[float(new['es'][ind[1]]),float(new['es'][ind[1]+1])],
        'S48_upper':float(new['S48'][ind]),'Q48_upper':float(new['Q48'][ind]),
        'bridge_integral_upper':float(new['bridge'][flat]),'tail_integral_upper':float(new['tail'][flat]),
        'adverse_budget_upper':float(E.hi),'paired_later_upper_recomputed':float(pair),'original_all_later_upper':float(new['later'][ind])}
    (out/'paired_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--blocks',nargs='+',type=int,default=list(range(9)));a=p.parse_args()
    if a.worker is not None:capture(a.worker);return
    start=time.time();dest=ROOT/'adaptive_paired_reports';dest.mkdir(exist_ok=True)
    def run(i):
        with (dest/f'block{i:02d}.log').open('w') as stream:r=subprocess.run([sys.executable,__file__,'--worker',str(i)],env=os.environ|{'OPENBLAS_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT)
        assert r.returncode==0,(i,r.returncode)
        return json.loads((dest/(f'block{i:02d}' if i<8 else 'refine07last')/'paired_report.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:reports=list(pool.map(run,a.blocks))
    result={'passed':all(r['all_original_arrays_bitwise_identical'] for r in reports),'blocks':reports,'seconds':time.time()-start}
    (dest/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
