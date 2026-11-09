"""Independent free-species reproduction, not importing archived ODE implementation.
State is free target RNA, bound RNA, free repressor, Output, Output integral.
Forward sensitivities use the directly derived free-species Jacobian.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
BASE=Path(__file__).resolve().parents[1]
REP=BASE/'repro'
fit=json.loads((REP/'committed_fit.json').read_text())
conf=json.loads((REP/'model_config.json').read_text())
p=conf['params'] | fit['parameters']['fitted_shared']
ic=conf['initial_conditions_RNA']

def constants(mech,q):
    suffix='model_1' if mech=='model_1' else 'model_2'
    gammaG=p['gamma']/p['length_output']; gammaR=p['gamma']/p['length_'+suffix]
    kG=(gammaG+p['kappa_minus'])/(p['kappa_plus']*p['length_output']/p['ribo_footprint_size'])
    kR=(gammaR+p['kappa_minus'])/(p['kappa_plus']*p['length_'+suffix]/p['ribo_footprint_size'])
    nu=fit['parameters']['nuisance_per_arm']['reference_'+mech+'_RNA']
    k0=ic['RNA_Burden']*nu['kappa_burden_circuit']*(q/20)**nu['p_dose_circuit']
    return dict(beta=p['beta'],extra=p['beta']*p['lambda_beta'] if mech=='model_2' else 0.,
                omega=p['theta'] if mech=='model_2' else 0.,off=p['eta_minus_'+suffix],e=p['eta_plus_'+suffix],
                d=p['delta']+p['mu'],kG=kG,kR=kR,gR=gammaR,vg=gammaG*p['total_rtl']/kG,
                R=p['total_rtl'],k0=k0,pdel=nu['p_dose_circuit'])

def solve(mech,q,eps,method='Radau',rtol=1e-10,scale=1.,sens=True,T=48.):
    c=constants(mech,q)
    beta,extra,omega,off,e,d=[c[k] for k in ['beta','extra','omega','off','e','d']]
    hb=beta+extra+off
    def fun(t,zz):
        m,bound,Z,Y,C=zz[:5]
        decay=np.exp(-beta*t)
        rr=ic['RNA_Repressor']/c['kR']*decay
        burden=c['k0']*scale/c['kG']*decay
        F=c['gR']*c['R']*rr
        a=m+omega*bound
        D=1+rr+burden+eps*a/c['kG']
        bind=e*Z*m
        f=np.array([off*bound-beta*m-bind,bind-hb*bound,
                    F/D+hb*bound-d*Z-bind,c['vg']*a/D-d*Y,Y])
        if not sens:return f
        U=eps*F/(c['kG']*D*D)
        Vm=c['vg']*(D-eps*a/c['kG'])/(D*D)
        J=np.array([[-beta-e*Z,off,-e*m,0,0],
                    [e*Z,-hb,e*m,0,0],
                    [-U-e*Z,hb-omega*U,-d-e*m,0,0],
                    [Vm,omega*Vm,0,-d,0],
                    [0,0,0,1,0]])
        forcing=np.array([0,0,-F*burden/(D*D),-c['vg']*a*burden/(D*D),0])
        return np.r_[f,J@zz[5:]+forcing]
    ini=np.zeros(10 if sens else 5);ini[0]=ic['RNA_Output']
    def jac(t,zz):
        zz=np.asarray(zz,dtype=complex)
        return np.column_stack([np.imag(fun(t,zz+1e-30j*np.eye(len(zz))[i]))/1e-30 for i in range(len(zz))])
    sol=solve_ivp(fun,(0,T),ini,method=method,rtol=rtol,atol=rtol/100,dense_output=True,jac=jac)
    assert sol.success,sol.message
    return sol,c

def maximum(f,T=48):
    ts=np.linspace(0,T,3001);v=f(ts)
    candidates=[(float(v[0]),0.),(float(v[-1]),T)]
    for i in np.flatnonzero((v[1:-1]>v[:-2])&(v[1:-1]>=v[2:]))+1:
        o=minimize_scalar(lambda t:-float(f(t)),bounds=(ts[i-1],ts[i+1]),method='bounded',options={'xatol':2e-11})
        candidates.append((-o.fun,o.x))
    return max(candidates)

def metrics(ref,loaded):
    peak0,tp0=maximum(lambda t:ref.sol(t)[3]);peak,tp=maximum(lambda t:loaded.sol(t)[3])
    diff,td=maximum(lambda t:np.abs(loaded.sol(t)[3]-ref.sol(t)[3]))
    return {'Dinf':diff/peak0,'E':loaded.sol(48)[4]/ref.sol(48)[4]-1,
            'R':1-loaded.sol(48)[9]/ref.sol(48)[9],
            'peak_delay_h':tp-tp0,'peak_relative_change':peak/peak0-1,
            'endpoint_relative_change':loaded.sol(48)[3]/ref.sol(48)[3]-1,
            'reference_peak_time':tp0,'modified_peak_time':tp,
            'C24_loaded':loaded.sol(24)[4],'C48_loaded':loaded.sol(48)[4],
            'S24_loaded_theta':loaded.sol(24)[9],'S48_loaded_theta':loaded.sol(48)[9],
            'Q24_loaded_theta':loaded.sol(24)[8]}

def main():
    start=time.time();rows=[];checks=[]
    for mech in ['model_1','model_2']:
      for q in [20.,50.,100.]:
        cases={}
        for eps in [0.,1.]:
          sol,c=solve(mech,q,eps)
          other,_=solve(mech,q,eps,method='BDF',rtol=5e-12)
          step=1e-4
          plus,_=solve(mech,q,eps,scale=np.exp(step),sens=False,rtol=1e-11)
          minus,_=solve(mech,q,eps,scale=np.exp(-step),sens=False,rtol=1e-11)
          sampled=np.linspace(0,48,961)
          relative=np.max(np.abs(sol.sol(sampled)-other.sol(sampled))/np.maximum(1,np.abs(sol.sol(sampled))))
          errs=[]
          for T in [24.,48.]:
            fd=(plus.sol(T)[4]-minus.sol(T)[4])/(2*step)
            errs.append(abs(fd-sol.sol(T)[9])/max(1,abs(sol.sol(T)[9])))
          checks.append({'mechanism':mech,'dose_ng':q,'loading':eps,'solver_max_scaled_error':relative,
                         'sensitivity_fd_max_scaled_error':max(errs),'k0':c['k0'],'pdel':c['pdel'],
                         'min_species':float(sol.sol(sampled)[:4].min()),
                         'S24_theta':float(sol.sol(24)[9]),'S48_theta':float(sol.sol(48)[9]),'Q24_theta':float(sol.sol(24)[8])})
          cases[eps]=sol
        row={'mechanism':mech,'dose_ng':q,**metrics(cases[0.],cases[1.])}
        rows.append(row)
        print(json.dumps(row),flush=True)
        Path(__file__).with_name('repression_independent_checkpoint.json').write_text(json.dumps({'rows':rows,'checks':checks},indent=2))
    # archived numerical diagnostic only; compare, do not import the old implementation.
    oldL=json.loads((BASE/'results/design_assessment.json').read_text())['rows']
    oldM=json.loads((BASE/'results/model_2_assessment.json').read_text())['RNA_rows']
    for row in rows:
      old=next(r for r in (oldL if row['mechanism']=='model_1' else oldM) if r['dose_ng']==row['dose_ng'] and r['loading']==1 and (r.get('cell','reference')=='reference'))
      row['max_abs_difference_from_archived_four_dimensionless_metrics']=max(abs(row[k]-old[oldkey]) for k,oldkey in [('Dinf','Dinf'),('E','E'),('R','R'),('endpoint_relative_change','endpoint_relative_change')])
      row['peak_delay_difference_from_archive_h']=row['peak_delay_h']-old['peak_time_shift']
    out={'scope':'Fresh independent free-species ODE and sensitivity reproduction at a fixed archived parameter set; diagnostics, not interval certification or experimental validation.',
         'rows':rows,'checks':checks,'seconds':time.time()-start,
         'max_solver_scaled_error':max(v['solver_max_scaled_error'] for v in checks),
         'max_fd_scaled_error':max(v['sensitivity_fd_max_scaled_error'] for v in checks),
         'provenance':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [REP/'committed_fit.json',REP/'model_config.json']}}
    Path(__file__).with_name('repression_independent_validation.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','checks']},indent=2),flush=True)
if __name__=='__main__':main()
