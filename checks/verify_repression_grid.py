from verify_repression_independent import *
def main():
 start=time.time();rows=[]
 oldL=json.loads((BASE/'results/design_assessment.json').read_text())['rows']
 oldM=json.loads((BASE/'results/model_2_assessment.json').read_text())['RNA_rows']
 times=np.r_[np.geomspace(.01,1,41),np.linspace(1,48,189)]
 for mech in ['model_1','model_2']:
  for q in [20.,50.,100.]:
   ref,_=solve(mech,q,0,rtol=2e-10)
   for eps in np.linspace(0,1,9):
    ss=ref if eps==0 else solve(mech,q,float(eps),rtol=2e-10)[0]
    met=metrics(ref,ss)
    old=next(r for r in (oldL if mech=='model_1' else oldM) if r['dose_ng']==q and r['loading']==eps and r.get('cell','reference')=='reference')
    err=max(abs(met[k]-old[k]) for k in ['Dinf','E','R','endpoint_relative_change'])
    rows.append(dict(mechanism=mech,dose_ng=q,loading=eps,max_archive_dimensionless_error=err,max_sampled_S_theta=float(ss.sol(times)[9].max()),Dinf=met['Dinf'],E=met['E'],R=met['R']))
   print(mech,q,'complete',flush=True)
 out=dict(scope='Independent sampled diagnostic only, no continuous-domain or all-horizon certificate.',rows=rows,seconds=time.time()-start,max_archive_error=max(r['max_archive_dimensionless_error'] for r in rows),max_sampled_S_theta=max(r['max_sampled_S_theta'] for r in rows),times=times.tolist())
 Path(__file__).with_name('repression_grid_independent_validation.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ['rows','times']},indent=2))
if __name__=='__main__':main()
