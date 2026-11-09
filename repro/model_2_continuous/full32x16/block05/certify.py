"""Outward-rounded certificate for the fixed-fit RNA model_1 model.

Ordinary ODE solves propose continuous Hermite polynomials. Bernstein residual
bounds certify the corner trajectories. A positive comparison system then
covers every point in each dose/load rectangle; no sampled slope is used.
"""
import json,math,sys,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).parent
DOWN=lambda a:np.nextafter(a,-np.inf)
UP=lambda a:np.nextafter(a,np.inf)

@dataclass
class I:
    lo:object
    hi:object=None
    def __post_init__(self):
        self.lo=np.asarray(self.lo,dtype=float)
        self.hi=self.lo.copy() if self.hi is None else np.asarray(self.hi,dtype=float)
    def __add__(self,o):
        o=iv(o);return I(DOWN(self.lo+o.lo),UP(self.hi+o.hi))
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,o):return self+-iv(o)
    def __rsub__(self,o):return iv(o)+-self
    def __mul__(self,o):
        o=iv(o);p=np.array([self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi])
        return I(DOWN(np.min(p,axis=0)),UP(np.max(p,axis=0)))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=iv(o)
        if np.any((o.lo<=0)&(o.hi>=0)):raise ValueError('interval division by zero')
        return self*I(DOWN(1/o.hi),UP(1/o.lo))
    def __rtruediv__(self,o):return iv(o)/self
    def __getitem__(self,k):return I(self.lo[k],self.hi[k])
    def absmax(self):return np.maximum(np.abs(self.lo),np.abs(self.hi))

def iv(o):return o if isinstance(o,I) else I(o)
def raw(v):return I(DOWN(float(v)),UP(float(v)))
def istack(xs,axis=-1):return I(np.stack([iv(x).lo for x in xs],axis),np.stack([iv(x).hi for x in xs],axis))

def exp_pos(x):
    """Taylor/geometric enclosure of exp(x), for x >= 0.

    Scaling to x/16 and four squarings avoids cancellation. No platform exp
    implementation is trusted by the certificate.
    """
    x=iv(x)
    tiny_negative=x.lo<0
    if np.any(x.lo < -1e-300):raise ValueError('positive exponential required')
    x=I(np.maximum(x.lo,0),x.hi)
    if np.any(x.hi>32):raise ValueError('exp argument beyond certified range')
    y=x/16;term=I(np.ones_like(x.lo));s=term
    for n in range(1,33):
        term=term*y/n;s=s+term
    nxt=term*y/33
    tail=(nxt/(1-y/34)).hi
    out=I(s.lo,UP(s.hi+tail))
    for _ in range(4):out=out*out
    if np.any(tiny_negative):out.lo=np.where(tiny_negative,DOWN(np.minimum(out.lo,1.0)),out.lo)
    return out

def exp_neg(x):return 1/exp_pos(iv(x))

def elevate(b,n):
    m=b.lo.shape[-1]-1
    if m==n:return b
    xs=[]
    for k in range(n+1):
        s=I(np.zeros(b.lo.shape[:-1]))
        for j in range(max(0,k-(n-m)),min(m,k)+1):
            w=I(math.comb(m,j))*math.comb(n-m,k-j)/math.comb(n,k)
            s=s+w*b[...,j]
        xs.append(s)
    return istack(xs)

def padd(a,b):
    n=max(a.lo.shape[-1],b.lo.shape[-1])-1
    return elevate(a,n)+elevate(b,n)

def pscale(b,c):return b*iv(c)[...,None]

def pmul(a,b):
    m=a.lo.shape[-1]-1;n=b.lo.shape[-1]-1;xs=[]
    for k in range(m+n+1):
        s=I(np.zeros(np.broadcast_shapes(a.lo.shape[:-1],b.lo.shape[:-1])))
        for j in range(max(0,k-n),min(m,k)+1):
            w=I(math.comb(m,j))*math.comb(n,k-j)/math.comb(m+n,k)
            s=s+w*a[...,j]*b[...,k-j]
        xs.append(s)
    return istack(xs)

def pdiff(b,dt):
    n=b.lo.shape[-1]-1
    return pscale(b[...,1:]-b[...,:-1],n/iv(dt))

def prange(b):return I(np.min(b.lo,axis=-1),np.max(b.hi,axis=-1))

def load_parameters():
    path=ROOT/'committed_fit.json'
    if path.exists():
        fit=json.loads(path.read_text());base=json.loads((ROOT/'model_config.json').read_text())
    else:
        fit=json.loads((ROOT/'source/fitting/campaign/phase3_perarm/best_M0_perarm.json').read_text())
        base=json.loads((ROOT/'source/models/Repressor_iFFL.json').read_text())
    p=base['params']|fit['parameters']['fitted_shared']
    v={k:raw(x) for k,x in p.items()}
    v['M0']=raw(base['initial_conditions_RNA']['RNA_Output'])
    v['r0']=raw(base['initial_conditions_RNA']['RNA_Repressor'])
    v['kbase']=raw(base['initial_conditions_RNA']['RNA_Burden'])
    v['e']=v['eta_plus_model_1'];v['h']=v['beta']+v['eta_minus_model_1']
    v['d']=v['delta']+v['mu'];v['R']=v['total_rtl']
    v['gR']=v['gamma']/v['length_model_1'];v['gG']=v['gamma']/v['length_output']
    v['kG']=(v['gG']+v['kappa_minus'])/(v['kappa_plus']*v['length_output']/v['ribo_footprint_size'])
    v['kR']=(v['gR']+v['kappa_minus'])/(v['kappa_plus']*v['length_model_1']/v['ribo_footprint_size'])
    v['vc']=v['gG']*v['R']/v['kG']
    return p,v,fit,base

def jets(x,ts,ks,es,nom):
    """Taylor coefficients through order three, evaluated at supplied states."""
    beta,e,h,d,M0,r0,kG,kR,gR,R=nom
    k=ks[:,None,None];ep=es[None,:,None]
    shape=x.shape[:-1];cs=np.zeros(shape+(4,));zs=cs.copy()
    cs[...,0]=x[...,0];zs[...,0]=x[...,1]
    E=np.exp(-beta*ts)[None,None,:]
    Eser=[E*(-beta)**j/math.factorial(j) for j in range(4)]
    for n in range(3):
        ds=[];fs=[]
        for j in range(n+1):
            mj=M0*Eser[j]-cs[...,j]
            ds.append((1 if j==0 else 0)+(r0/kR+k/kG)*Eser[j]+ep*mj/kG)
            fs.append(gR*R*r0/kR*Eser[j])
        us=[]
        for j in range(n+1):
            val=fs[j].copy()+np.zeros_like(ds[0])
            for q in range(1,j+1):val=val-ds[q]*us[j-q]
            us.append(val/ds[0])
        bind=-h*cs[...,n]
        for j in range(n+1):bind+=e*zs[...,j]*(M0*Eser[n-j]-cs[...,n-j])
        cs[...,n+1]=bind/(n+1)
        zs[...,n+1]=(us[n]-d*zs[...,n]-bind)/(n+1)
    return cs,zs

def hermite7(jets,dt):
    out=np.zeros(jets.shape[:-2]+(jets.shape[-2]-1,8))
    for i in range(4):
        for j in range(i+1):
            w=math.comb(i,j)/math.comb(7,j)
            out[...,i]+=w*jets[...,:-1,j]*dt**j
            out[...,7-i]+=w*jets[...,1:,j]*(-dt)**j
    return out

def delivery_decimal_bracket(line):
    """Outward Decimal bounds for the exact JSON delivery endpoint literals.

    Decimal ln/exp are correctly rounded to nearest (half-even). One adjacent
    Decimal on each side therefore brackets their exact result; the same
    widening encloses every rounded product. The delivery exponent is positive.
    """
    from decimal import Decimal,localcontext,ROUND_HALF_EVEN
    fit_path=ROOT/'committed_fit.json'
    base_path=ROOT/'model_config.json'
    if not fit_path.exists():
        fit_path=ROOT/'source/fitting/campaign/phase3_perarm/best_M0_perarm.json'
        base_path=ROOT/'source/models/Repressor_iFFL.json'
    exact_fit=json.loads(fit_path.read_text(),parse_float=Decimal,parse_int=Decimal)
    exact_base=json.loads(base_path.read_text(),parse_float=Decimal,parse_int=Decimal)
    nu=exact_fit['parameters']['nuisance_per_arm'][line+'_model_1_RNA']
    m=exact_base['initial_conditions_RNA']['RNA_Burden']
    kappa=nu['kappa_burden_circuit'];power=nu['p_dose_circuit']
    if not all(x.is_finite() and x>0 for x in (m,kappa,power)):
        raise AssertionError('positive finite delivery literals required')
    with localcontext() as ctx:
        ctx.prec=70;ctx.rounding=ROUND_HALF_EVEN
        ctx.Emin=-999999;ctx.Emax=999999;ctx.clamp=0
        low=m*kappa
        low_lo=low.next_minus();low_hi=low.next_plus()
        log5=Decimal(5).ln()
        arg_lo=(log5.next_minus()*power).next_minus()
        arg_hi=(log5.next_plus()*power).next_plus()
        exp_lo=arg_lo.exp().next_minus()
        exp_hi=arg_hi.exp().next_plus()
        high_lo=(low_lo*exp_lo).next_minus()
        high_hi=(low_hi*exp_hi).next_plus()
        if not all(x.is_finite() and x>0 for x in (low_lo,low_hi,high_lo,high_hi)):
            raise AssertionError('delivery bracket outside supported Decimal range')
    return low_lo,high_hi


def delivery_float_cover(line):
    """Binary64 exterior endpoints, checked against exact Decimal conversion."""
    from decimal import Decimal
    low,high=delivery_decimal_bracket(line)
    kl=float(low);ku=float(high)
    if not (math.isfinite(kl) and math.isfinite(ku) and kl>0 and ku>0):
        raise AssertionError('delivery cover outside supported binary64 range')
    while Decimal.from_float(kl)>low:kl=float(DOWN(kl))
    while Decimal.from_float(ku)<high:ku=float(UP(ku))
    if not (0<kl<ku and math.isfinite(ku)):
        raise AssertionError('delivery interval has no positive finite binary64 exterior cover')
    return kl,ku


def prepare_data(line='reference',nd=32,ne=8):
    """Creates polynomial proposals; the certificate subsequently checks them."""
    out=ROOT/f'corner_polynomials_{line}.npz'
    if out.exists():return out
    p,v,fit,base=load_parameters()
    nom=[p['beta'],p['eta_plus_model_1'],p['beta']+p['eta_minus_model_1'],p['delta']+p['mu'],
         base['initial_conditions_RNA']['RNA_Output'],base['initial_conditions_RNA']['RNA_Repressor'],
         (p['gamma']/p['length_output']+p['kappa_minus'])/(p['kappa_plus']*p['length_output']/p['ribo_footprint_size']),
         (p['gamma']/p['length_model_1']+p['kappa_minus'])/(p['kappa_plus']*p['length_model_1']/p['ribo_footprint_size']),
         p['gamma']/p['length_model_1'],p['total_rtl']]
    beta,e,h,d,M0,r0,kG,kR,gR,R=nom
    kl,ku=delivery_float_cover(line)
    ks=np.geomspace(kl,ku,nd+1);ks[0]=kl;ks[-1]=ku;es=np.linspace(0,1,ne+1)
    ts=np.concatenate([np.linspace(0,.2,101),np.linspace(.2,48,2391)[1:]])
    # Insert exact 24-hour boundary without shifting any existing time points.
    ts=np.unique(np.r_[ts,24.])
    x=np.empty((nd+1,ne+1,len(ts),2))
    for i,k0 in enumerate(ks):
        for j,ep in enumerate(es):
            def rhs(t,w):
                c,Z=w;Ex=math.exp(-beta*t);M=M0*Ex;rr=r0/kR*Ex;B=1+rr+k0/kG*Ex
                F=gR*R*rr;bind=e*Z*(M-c)-h*c
                return bind,F/(B+ep*(M-c)/kG)-d*Z-bind
            sol=solve_ivp(rhs,(0,48),(0,0),method='Radau',rtol=2e-12,atol=2e-12,dense_output=True)
            if not sol.success:raise RuntimeError(sol.message)
            x[i,j]=sol.sol(ts).T
        if i%4==0:print(json.dumps({'phase':'polynomial_proposals','line':line,'row':i,'total':nd+1}),flush=True)
    jc,jz=jets(x,ts,ks,es,nom)
    cp=hermite7(jc,np.diff(ts));zp=hermite7(jz,np.diff(ts))
    np.savez_compressed(out,ts=ts,ks=ks,es=es,cp=cp,zp=zp)
    return out

def exp_polynomial(beta,starts,steps,n=10):
    base=exp_neg(beta*I(starts));bh=beta*iv(steps)
    power=[base]
    for j in range(1,n+1):power.append(-power[-1]*bh/j)
    coeff=[]
    for i in range(n+1):
        s=I(np.zeros_like(starts))
        for j in range(i+1):s=s+power[j]*(I(math.comb(i,j))/math.comb(n,j))
        coeff.append(s)
    b=istack(coeff)
    rem=base*exp_pos(bh)
    for j in range(1,n+2):rem=rem*bh/j
    return I(DOWN(b.lo-rem.hi[...,None]),UP(b.hi+rem.hi[...,None]))

def state_bounds(data,v):
    ts=data['ts'];ks=data['ks'];es=data['es'];cp=data['cp'];zp=data['zp']
    # A cache is a proposal, not trusted proof data: validate the assumptions
    # needed by the residual tube and the finite rectangle cover.
    if not (ts[0]==0 and ts[-1]==48 and 24 in ts and np.all(np.diff(ts)>0)):
        raise AssertionError('time cover must run continuously from 0 to 48 with a 24 h boundary')
    if not (ks[0]>0 and np.all(np.diff(ks)>0) and es[0]==0 and es[-1]==1 and np.all(np.diff(es)>0)):
        raise AssertionError('invalid dose/loading cover')
    shape=(len(ks),len(es),len(ts)-1,8)
    if cp.shape!=shape or zp.shape!=shape or not np.all(np.isfinite(cp)) or not np.all(np.isfinite(zp)):
        raise AssertionError('invalid polynomial proposals')
    for coeff in [cp,zp]:
        if not np.all(coeff[...,0,0]==0) or not np.array_equal(coeff[...,:-1,7],coeff[...,1:,0]):
            raise AssertionError('polynomial proposals must have exact initial zero and continuous joins')
    dt=I(DOWN(np.diff(ts)),UP(np.diff(ts)))
    # Time nodes and polynomial coefficients are exact binary64 numbers.
    # Derivatives use enclosures of their exact mathematical differences.
    E=exp_polynomial(v['beta'],ts[:-1],dt)
    cc=I(cp);zz=I(zp)
    cb=pdiff(cc,dt);zb=pdiff(zz,dt)
    Mp=pscale(E,v['M0'])
    Mp=I(Mp.lo[None,None,...],Mp.hi[None,None,...])
    # Domain checks: all control points certify c >= 0, Z >= 0, M-c >= 0.
    fre=padd(Mp,-cc)
    if np.min(cc.lo)<0 or np.min(zz.lo)<0 or np.min(fre.lo)<0:
        raise AssertionError(('proposal leaves physical domain',float(cc.lo.min()),float(zz.lo.min()),float(fre.lo.min())))
    rc=padd(padd(cb,pscale(cc,v['h'])),pscale(pmul(zz,cc),v['e']))
    rc=padd(rc,-pscale(pmul(zz,Mp),v['e']))
    total=padd(padd(cb,zb),pscale(zz,v['d']))
    loads=v['r0']/v['kR']+I(ks[:,None,None])/v['kG']
    Bp=padd(I(np.ones((len(ks),1,1,1))),pscale(I(E.lo[None,None,...],E.hi[None,None,...]),loads))
    Dp=padd(Bp,pscale(fre,I(es[None,:,None])/v['kG']))
    Fp=pscale(I(E.lo[None,None,...],E.hi[None,None,...]),v['gR']*v['R']*v['r0']/v['kR'])
    num=padd(pmul(total,Dp),-Fp)
    Dlo=prange(Dp).lo
    rho=(2*I(prange(rc).absmax())+I(prange(num).absmax())/I(Dlo)).hi
    rho_corner=np.max(rho,axis=-1)
    G=(I(es[None,:])*v['gR']*v['R']/(4*v['kG']*(1+I(ks[:,None])*v['kR']/(v['r0']*v['kG'])))).hi
    err=(I(rho_corner)*I(ts[-1])*exp_pos(I(G)*I(ts[-1]))).hi
    cr=prange(cc);zr=prange(zz)
    return cr,zr,err,rho_corner,G

def pos_exp_action(P,H,a,b,c,f,dt):
    """Upper enclosure of exact constant Metzler affine flow.

    Uniformization with a positive Taylor series and geometric tail; the
    factor exp(-q*dt) is bounded above by reciprocal partial exp series.
    """
    tau=iv(dt)
    q=np.maximum(a,float(DPAR))
    aa=(I(q)-I(a)).hi;bb=b;cc=c;hh=(I(q)-I(DPAR)).hi
    pp=P.copy();zz=H.copy();one=np.ones_like(P)
    sP=I(P);sH=I(H)
    for n in range(1,41):
        pnew=((I(aa)*I(pp)+I(bb)*I(zz))*tau/n).hi
        hnew=((I(cc)*I(pp)+I(hh)*I(zz)+I(f)*I(one))*tau/n).hi
        one=(I(one)*I(q)*tau/n).hi
        pp,zz=pnew,hnew;sP=sP+I(pp);sH=sH+I(zz)
    normN=np.maximum(np.maximum((I(aa)+I(bb)).hi,(I(cc)+I(hh)+I(f)).hi),q)
    s=(I(normN)*tau).hi
    if np.any(s>=42):raise ValueError('uniformization series tail not geometric')
    tail=I(np.maximum(np.maximum(P,H),1))
    for n in range(1,42):tail=tail*I(s)/n
    tail=tail/(1-I(s)/42)
    scale=exp_neg(I(q)*tau)
    return ((sP+tail)*scale).hi,((sH+tail)*scale).hi

def certify(line='reference'):
    started=time.time();p,v,fit,base=load_parameters()
    data=np.load(prepare_data(line));ts=data['ts'];ks=data['ks'];es=data['es']
    from decimal import Decimal
    low,high=delivery_decimal_bracket(line)
    if Decimal.from_float(float(ks[0]))>low or Decimal.from_float(float(ks[-1]))<high:
        raise AssertionError('cached dose cover excludes the committed delivery interval')
    cr,zr,E,rho,G=state_bounds(data,v)
    print(json.dumps({'phase':'validated_corner_tubes','line':line,'max_residual':float(rho.max()),'max_state_error':float(E.max())}),flush=True)
    # Cooperative order puts all states between opposite parameter corners.
    c_high=UP(cr.hi[:-1,:-1,:]+E[:-1,:-1,None]).reshape(-1,len(ts)-1)
    c_low=DOWN(cr.lo[1:,1:,:]-E[1:,1:,None]).reshape(-1,len(ts)-1)
    z_low=np.maximum(0,DOWN(zr.lo[1:,1:,:]-E[1:,1:,None])).reshape(-1,len(ts)-1)
    nd=len(ks)-1;ne=len(es)-1;nc=nd*ne
    klo=I(np.repeat(ks[:-1],ne));khi=I(np.repeat(ks[1:],ne))
    elo=I(np.tile(es[:-1],nd));ehi=I(np.tile(es[1:],nd))
    P=np.zeros(nc);H=np.zeros(nc);out={24:np.zeros(nc),48:np.zeros(nc)}
    endpoint_upper=np.zeros(nc)
    endpoint_kernel=exp_neg(v['d']*(I(48)-I(ts)))
    Qrec=np.zeros(nc);Srec=np.zeros(nc)
    all_time_ok=np.ones(nc,dtype=bool);first_forcing_upper=None
    largest_finite_cell_cap=-np.inf
    lam={}
    for T in out:
        lag=I(T)-I(ts)
        lag=I(np.maximum(lag.lo,0),np.maximum(lag.hi,0))
        lam[T]=(1-exp_neg(v['d']*lag))/v['d']
    global DPAR
    DPAR=float(v['d'].lo)
    for it,t in enumerate(ts[:-1]):
        te=ts[it+1];delta=I(te)-I(t);dthi=float(delta.hi)
        ex=exp_neg(v['beta']*I(t,te));M=v['M0']*ex
        ml=np.maximum(0,(I(M.lo)-I(c_high[:,it])).lo)
        mu=np.minimum(float(M.hi),(I(M.hi)-I(c_low[:,it])).hi)
        r=v['r0']/v['kR']*ex
        Bl=(1+r+klo/v['kG']*ex).lo;Bu=(1+r+khi/v['kG']*ex).hi
        bl=(klo/v['kG']*ex).lo;bu=(khi/v['kG']*ex).hi
        Fu=(v['gR']*v['R']*r).hi
        Dl=(I(Bl)+elo*I(ml)/v['kG']).lo;Du=(I(Bu)+ehi*I(mu)/v['kG']).hi
        a=(v['e']*I(z_low[:,it])+v['h']+v['e']*I(ml)).lo
        b=(v['e']*I(mu)).hi
        c=(v['d']+ehi*I(Fu)/(v['kG']*I(Dl)*I(Dl))).hi
        f=(I(Fu)*I(bu)/(I(Dl)*I(Dl))).hi
        gam=np.maximum(np.maximum((I(b)-I(a)).hi,(I(c)-I(DPAR)).hi),0)
        Hsup=(exp_pos(I(gam)*delta)*(I(np.maximum(P,H))+I(f)*delta)).hi
        Psup=np.maximum(P,(I(b)*I(Hsup)/I(a)).hi)
        # A real-valued output-sensitivity upper flow. Its forcing is a
        # constant upper bound throughout this time cell; signed interval
        # arithmetic retains both negative and positive contributions.
        Vupper=(v['vc']*(I(Bu)*I(Psup)/(I(Dl)*I(Dl))-I(ml)*I(bl)/(I(Du)*I(Du)))).hi
        decay=exp_neg(v['d']*delta);weight=(1-decay)/v['d']
        Qnext=(decay*I(Qrec)+weight*I(Vupper)).hi
        Qsup=np.maximum(Qrec,Qnext)
        if it==0:
            # Starting with S=Q=0, a strictly negative forcing proves
            # S(T)<0 for every interior T>0 of the first cell.
            first_forcing_upper=Vupper.copy()
            all_time_ok &= Vupper<0
        else:
            cap=(I(Srec)+delta*I(np.maximum(Qsup,0))).hi
            all_time_ok &= cap<0
            largest_finite_cell_cap=max(largest_finite_cell_cap,float(cap.max()))
        Srec=(I(Srec)+weight*I(Qrec)+(delta-weight)*I(Vupper)/v['d']).hi
        Qrec=Qnext
        for T in out:
            if te<=T:
                pos=I(lam[T].hi[it])*I(Bu)*I(Psup)/(I(Dl)*I(Dl))
                neg=I(lam[T].lo[it+1])*I(ml)*I(bl)/(I(Du)*I(Du))
                contribution=delta*v['vc']*(pos-neg)
                out[T]=(I(out[T])+I(contribution.hi)).hi
        P,H=pos_exp_action(P,H,a,b,c,f,delta)
        endpoint_pos=I(endpoint_kernel.hi[it+1])*I(Bu)*I(Psup)/(I(Dl)*I(Dl))
        endpoint_neg=I(endpoint_kernel.lo[it])*I(ml)*I(bl)/(I(Du)*I(Du))
        endpoint_upper=(I(endpoint_upper)+delta*v['vc']*(endpoint_pos-endpoint_neg)).hi
        if not np.all(np.isfinite(P+H)):raise AssertionError('nonfinite comparison')
    result={'line':line,'dose_ng':[20,100],'coupling_interval':[0,1],'dose_cells':nd,'coupling_cells':ne,
            'time_cells':len(ts)-1,'time_end':float(ts[-1]),'max_corner_residual_l1':float(rho.max()),
            'max_corner_state_error_l1':float(E.max()),'max_log_norm_gain':float(G.max()),
            'upper_cumulative_log_burden_slope':{str(T):float(out[T].max()) for T in out},
            'all_cells_strictly_negative':{str(T):bool(np.all(out[T]<0)) for T in out},
            'cell_upper_bounds':{str(T):out[T].reshape(nd,ne).tolist() for T in out},
            'all_positive_horizons_through_48':bool(np.all(all_time_ok)),
            'max_first_cell_production_slope_upper':float(first_forcing_upper.max()),
            'max_finite_cell_interior_slope_cap':largest_finite_cell_cap,
            'parameters':{k:[float(v[k].lo),float(v[k].hi)] for k in ['beta','e','h','d','gR','gG','kG','kR','R','M0','r0']},
            'nuisance':fit['parameters']['nuisance_per_arm'][line+'_model_1_RNA'],
            'seconds':time.time()-started,
            'method':'Bernstein residual tubes + outward-rounded positive comparison + kernel rectangle upper sum'}
    (ROOT/f'certificate_{line}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/f'certificate_cells_{line}.npz',ks=ks,es=es,upper24=out[24].reshape(nd,ne),upper48=out[48].reshape(nd,ne))
    np.savez_compressed(ROOT/f'certified_terminal_{line}.npz',ks=ks,es=es,P=P,H=H,Qupper=np.minimum(endpoint_upper,Qrec),Supper=np.minimum(out[48],Srec),all_time_ok=all_time_ok)
    print(json.dumps({k:x for k,x in result.items() if k not in ['cell_upper_bounds','parameters','nuisance']}),flush=True)
    return result

if __name__=='__main__':certify(sys.argv[1] if len(sys.argv)>1 else 'reference')
