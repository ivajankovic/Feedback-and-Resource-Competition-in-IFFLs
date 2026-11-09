"""Bilinear parameter / seventh-degree time residual proposals for model_2.

The proposal may be inaccurate: only its outward-rounded residual and joins
are used to enclose the exact ODE on each entire dose/loading rectangle.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
import certify as c

I, UP, DOWN = c.I, c.UP, c.DOWN


def move(x, a, b):
    return I(np.moveaxis(x.lo, a, b), np.moveaxis(x.hi, a, b))


def elevate(x, degrees):
    for axis, degree in zip([-3, -2, -1], degrees):
        x = move(c.elevate(move(x, axis, -1), degree), -1, axis)
    return x


def add(x, y):
    degrees = [max(a, b) - 1 for a, b in zip(x.lo.shape[-3:], y.lo.shape[-3:])]
    return elevate(x, degrees) + elevate(y, degrees)


def mul_slow(x, y):
    """Tensor-product Bernstein multiplication, with enclosed exact weights."""
    import math
    a, b = np.array(x.lo.shape[-3:]) - 1, np.array(y.lo.shape[-3:]) - 1
    output = []
    for k in range(a[0] + b[0] + 1):
        erows = []
        for l in range(a[1] + b[1] + 1):
            z = I(np.zeros(np.broadcast_shapes(x.lo.shape[:-3], y.lo.shape[:-3]) + (a[2] + b[2] + 1,)))
            for i in range(max(0, k-b[0]), min(a[0], k)+1):
                for j in range(max(0, l-b[1]), min(a[1], l)+1):
                    wk = I(math.comb(a[0], i)) * math.comb(b[0], k-i) / math.comb(a[0]+b[0], k)
                    we = I(math.comb(a[1], j)) * math.comb(b[1], l-j) / math.comb(a[1]+b[1], l)
                    z = z + c.pmul(x[..., i, j, :], y[..., k-i, l-j, :]) * (wk * we)
            erows.append(z)
        output.append(c.istack(erows, axis=-2))
    return c.istack(output, axis=-3)


def mul(x,y):
    """Same tensor Bernstein convolution; batch the time-polynomial products.

    All weights, products and sums use the existing outward interval primitives.
    Only the arrangement of independent NumPy vector operations changes.
    """
    import math
    a,b=np.array(x.lo.shape[-3:])-1,np.array(y.lo.shape[-3:])-1
    xa=I(x.lo.reshape(x.lo.shape[:-3]+((a[0]+1)*(a[1]+1),a[2]+1)),
         x.hi.reshape(x.hi.shape[:-3]+((a[0]+1)*(a[1]+1),a[2]+1)))
    ya=I(y.lo.reshape(y.lo.shape[:-3]+((b[0]+1)*(b[1]+1),b[2]+1)),
         y.hi.reshape(y.hi.shape[:-3]+((b[0]+1)*(b[1]+1),b[2]+1)))
    pairs=c.pmul(I(xa.lo[..., :,None,:],xa.hi[..., :,None,:]),
                 I(ya.lo[...,None,:,:],ya.hi[...,None,:,:]))
    output=[]
    prefix=pairs.lo.shape[:-3]
    for k in range(a[0]+b[0]+1):
        erows=[]
        for l in range(a[1]+b[1]+1):
            z=I(np.zeros(prefix+(a[2]+b[2]+1,)))
            for i in range(max(0,k-b[0]),min(a[0],k)+1):
                for j in range(max(0,l-b[1]),min(a[1],l)+1):
                    wk=I(math.comb(a[0],i))*math.comb(b[0],k-i)/math.comb(a[0]+b[0],k)
                    we=I(math.comb(a[1],j))*math.comb(b[1],l-j)/math.comb(a[1]+b[1],l)
                    z=z+pairs[...,i*(a[1]+1)+j,(k-i)*(b[1]+1)+(l-j),:]*(wk*we)
            erows.append(z)
        output.append(c.istack(erows,axis=-2))
    return c.istack(output,axis=-3)


def span(x):
    return I(x.lo.min(axis=(-3, -2, -1)), x.hi.max(axis=(-3, -2, -1)))


def prepare(root, cell, nd, ne, T1, cover, grid, jets, p):
    dk,de=3,2
    path = root / f"model_2_cubicquad_proposals_{cell}_{nd}_{ne}_{T1:g}.npz"
    if path.exists():
        return path
    kl, ku=cover(cell)
    ks=np.geomspace(kl,ku,nd+1);ks[0],ks[-1]=kl,ku
    es=np.linspace(0,1,ne+1)
    kn=np.concatenate([ks[:-1,None]+(ks[1:]-ks[:-1])[:,None]*np.arange(dk)[None,:]/dk],axis=0).ravel()
    kn=np.r_[kn,ks[-1]]
    en=np.linspace(0,1,ne*de+1)
    kc=np.repeat(kn,len(en));ec=np.tile(en,len(kn));ts=grid(T1)
    vals=[]
    for i,(k,ep) in enumerate(zip(kc,ec)):
        def rhs(t,x):
            M,cb,W=x;E=np.exp(-p.beta*t)
            B=1+(p.r0/p.kR+k/p.kG)*E;F=p.gR*p.R*p.r0/p.kR*E
            D=B+ep*(M-p.s*cb)/p.kG
            return [-p.beta*M-p.extra*cb,p.e*(W-cb)*(M-cb)-p.h*cb,F/D-p.d*(W-cb)]
        sol=solve_ivp(rhs,(0,T1),[p.M0,0.,0.],method="LSODA",t_eval=ts,rtol=2e-11,atol=2e-12)
        assert sol.success
        vals.append(sol.y.T)
        if (i+1)%32==0 or i+1==len(kc):print(json.dumps({"phase":"tensor_proposals","completed":i+1,"total":len(kc)}),flush=True)
    js=jets(np.asarray(vals),ts,kc,ec,p)
    poly=np.stack([c.hermite7(js[:,:,j,:],np.diff(ts)) for j in range(3)],axis=2)
    poly=poly.reshape(len(kn),len(en),len(ts)-1,3,8)
    import math
    nodes=np.linspace(0,1,dk+1)
    V=np.array([[math.comb(dk,j)*t**j*(1-t)**(dk-j) for j in range(dk+1)] for t in nodes])
    transform_k=np.linalg.inv(V)
    transform_e=np.array([[1.,0.,0.],[-.5,2.,-.5],[0.,0.,1.]])
    boxes=np.empty((nd,ne,len(ts)-1,3,dk+1,de+1,8))
    for i in range(nd):
        for j in range(ne):
            raw=poly[dk*i:dk*i+dk+1,de*j:de*j+de+1].transpose(2,3,0,1,4)
            boxes[i,j]=np.einsum('ai,bj,tsijk->tsabk',transform_k,transform_e,raw)
    np.savez_compressed(path,ks=ks,es=es,ts=ts,poly=boxes)
    return path


def component_radii(v, residual, mr, cr, wr, mpr, zpr, U, initial_M, dt, flow, action):
    """Self-consistent exponential barriers, followed by enclosed endpoint flow.

    Linear solves/eigenvalues propose positive barriers only. Every barrier is
    accepted by outward-rounded component inequalities on its own state tube.
    """
    from certify_model_2 import dot_interval
    nc, nt = mr.lo.shape
    x = np.zeros((nc,3))
    x[:,0] = (v["M0"]-initial_M).absmax()
    radii = np.empty((nc,nt,3))
    max_iterations=0
    for j in range(nt):
        rho=residual[:,j,:]
        guess=np.maximum(x*1.1+1e-9,1e-8)
        for iteration in range(30):
            def matrix(rad):
                em,ec,ew=[I(rad[:,k]) for k in range(3)]
                mj,zj=mpr[:,j],zpr[:,j]
                mlo=np.minimum(mj.lo,np.maximum(0,(mj-em-ec).lo))
                mhi=(mj+em+ec).hi
                zlo=np.minimum(zj.lo,np.maximum(0,(zj-ew-ec).lo))
                zhi=(zj+ew+ec).hi
                A=np.zeros((nc,3,3))
                A[:,0,0]=-float(v["beta"].lo);A[:,0,1]=float(v["extra"].hi)
                A[:,1,0]=(v["e"]*I(np.maximum(abs(zlo),abs(zhi)))).hi
                A[:,1,1]=-(v["h"]+v["e"]*(I(mlo,mhi)+I(zlo,zhi))).lo
                A[:,1,2]=(v["e"]*I(np.maximum(abs(mlo),abs(mhi)))).hi
                A[:,2,0]=U[:,j];A[:,2,1]=(v["d"]+v["s"]*I(U[:,j])).hi
                A[:,2,2]=-float(v["d"].lo)
                return A
            A=matrix(guess)
            eig=np.linalg.eigvals(A).real.max(axis=1)
            gamma=np.maximum(eig+.1,.1)
            shifted=gamma[:,None,None]*np.eye(3)-A
            q=np.linalg.solve(shifted,(rho+1e-10)[:,:,None])[:,:,0]
            w=np.linalg.solve(shifted,np.ones((nc,3,1)))[:,:,0]
            assert np.all(np.isfinite(q)) and np.all(q>=0) and np.all(w>0)
            kap=np.maximum(0,np.max((x-q)/w,axis=1))
            base=(q+kap[:,None]*w)*1.02+1e-10*w
            growth=c.exp_pos(I(gamma)*dt[j])
            candidate=(I(base)*I(growth.lo[:,None],growth.hi[:,None])).hi
            Acheck=matrix(candidate)
            lhs=dot_interval(Acheck,base[:,None,:])+I(rho)
            rhs=I(gamma[:,None])*I(base)
            good=np.all(lhs.hi<=rhs.lo,axis=1)&np.all(base>=x,axis=1)
            if np.all(good):
                radii[:,j]=candidate
                phi=flow(Acheck,rho,dt[j])
                x=action(phi,x)
                assert np.all(x<=candidate) and np.all(np.isfinite(x))
                max_iterations=max(max_iterations,iteration+1)
                break
            guess=np.maximum(candidate*1.3,guess*1.1)
        else:
            raise AssertionError(("state barrier did not validate",j,float(dt[j].hi)))
        if (j+1)%1000==0:
            print(json.dumps({"phase":"component_state_tubes","time_cell":j+1,
                              "max_error":float(radii[:,:j+1].max())}),flush=True)
    print(json.dumps({"phase":"component_state_tubes_complete","max_barrier_iterations":max_iterations}),flush=True)
    return radii


def enclose(data, v, chunk=4, flow=None, action=None):
    ts, ks, es, pp = [data[k] for k in ["ts", "ks", "es", "poly"]]
    nd, ne, nt = len(ks)-1, len(es)-1, len(ts)-1
    assert pp.shape == (nd, ne, nt, 3, 4, 3, 8) and np.all(np.isfinite(pp))
    assert ts[0] == 0 and ts[-1] >= 48 and 24 in ts and 48 in ts and np.all(np.diff(ts)>0)
    assert es[0] == 0 and es[-1] == 1 and np.all(np.diff(es)>0) and ks[0]>0 and np.all(np.diff(ks)>0)
    assert np.all(pp[:,:,0,1:,:,:,0] == 0) and np.array_equal(pp[:,:,:-1,:,:,:,7], pp[:,:,1:,:,:,:,0])
    dt = I(ts[1:])-I(ts[:-1])
    Ex = c.exp_polynomial(v["beta"], ts[:-1], dt)
    Ex = I(Ex.lo[None,:,None,None,:], Ex.hi[None,:,None,None,:])
    ii, jj = np.repeat(np.arange(nd),ne), np.tile(np.arange(ne),nd)
    collected = []
    for first in range(0, nd*ne, chunk):
        i, j = ii[first:first+chunk], jj[first:first+chunk]
        nc = len(i)
        pol = pp[i,j]
        # nc,time,state,k-degree,epsilon-degree,time-degree.
        M, cb, W = [I(pol[:,:,st]) for st in range(3)]
        mp, zp = add(M,-cb), add(W,-cb)
        a = add(M,-cb*v["s"])
        K = I(np.stack([ks[i],ks[i+1]],axis=-1)[:,None,:,None,None])
        ep = I(np.stack([es[j],es[j+1]],axis=-1)[:,None,None,:,None])
        B = add(I(np.ones((nc,1,1,1,1))), mul(add(I(np.ones((nc,1,1,1,1)))*v["r0"]/v["kR"], K/v["kG"]), Ex))
        D = add(B,mul(ep,a)/v["kG"])
        F = Ex*(v["gR"]*v["R"]*v["r0"]/v["kR"])
        def diff(x):
            return c.pdiff(x, I(dt.lo[None,:,None,None],dt.hi[None,:,None,None]))
        rm = add(add(diff(M),M*v["beta"]),cb*v["extra"])
        rc = add(add(diff(cb),cb*v["h"]),-mul(zp,mp)*v["e"])
        rw = add(mul(add(diff(W),zp*v["d"]),D),-F)
        Dlo = span(D).lo
        assert Dlo.min()>0
        residual = np.stack([span(rm).absmax(),span(rc).absmax(),(I(span(rw).absmax())/I(Dlo)).hi],axis=-1)
        mr, cr, wr, mpr, zpr, ar = [span(x) for x in [M,cb,W,mp,zp,a]]
        floor = (I(span(B).lo)+I(es[j+1,None])*I(np.minimum(ar.lo,0))/v["kG"]).lo
        assert floor.min()>0
        U = (I(es[j+1,None])*I(np.broadcast_to(span(F).hi,floor.shape))/(v["kG"]*I(floor)*I(floor))).hi
        bindmu = (-v["h"]+2*v["e"]*(I(np.maximum(0,-mpr.lo))+I(np.maximum(0,-zpr.lo)))).hi
        growth = np.maximum.reduce([np.full((nc,nt),float((v["extra"]-v["beta"]).hi)),bindmu,
                                   ((1+v["s"])*I(U)).hi,np.zeros((nc,nt))])
        initial_M=I(M.lo[:,0,:,:,0].min(axis=(1,2)),M.hi[:,0,:,:,0].max(axis=(1,2)))
        collected.append(dict(mrlo=mr.lo,mrhi=mr.hi,crlo=cr.lo,crhi=cr.hi,wrlo=wr.lo,wrhi=wr.hi,
                              mplo=mpr.lo,mphi=mpr.hi,zplo=zpr.lo,zphi=zpr.hi,alo=ar.lo,ahi=ar.hi,
                              U=U,residual=residual,growth_rate=growth,initial_M_lo=initial_M.lo,initial_M_hi=initial_M.hi))
        print(json.dumps({"phase":"tensor_residual_tubes","completed":first+nc,"total":nd*ne,
                          "max_residual":float(residual.max())}),flush=True)
    z={key:np.concatenate([block[key] for block in collected]) for key in collected[0]}
    mr,cr,wr,mpr,zpr,ar=[I(z[lo],z[hi]) for lo,hi in [("mrlo","mrhi"),("crlo","crhi"),("wrlo","wrhi"),
                                                       ("mplo","mphi"),("zplo","zphi"),("alo","ahi")]]
    radii=component_radii(v,z["residual"],mr,cr,wr,mpr,zpr,z["U"],I(z["initial_M_lo"],z["initial_M_hi"]),dt,flow,action)
    erM,erc,erW=[I(radii[:,:,i]) for i in range(3)]
    GU=(v["M0"]*c.exp_neg(v["beta"]*I(ts[:-1]))).hi
    GL=(v["M0"]*c.exp_neg((v["beta"]+v["extra"])*I(ts[1:]))).lo
    Ml=np.maximum.reduce([np.zeros_like(erM.lo),(mr-erM).lo,np.broadcast_to(GL,erM.lo.shape)])
    Mu=np.minimum((mr+erM).hi,GU)
    ml=np.maximum(0,(mpr-erM-erc).lo);mu=np.maximum(0,np.minimum(Mu,(mpr+erM+erc).hi))
    zl=np.maximum(0,(zpr-erW-erc).lo);zu=np.maximum(0,(zpr+erW+erc).hi)
    al=np.maximum.reduce([np.zeros_like(Ml),(ar-erM-v["s"]*erc).lo,(v["omega"]*I(Ml)).lo])
    au=np.minimum(Mu,(ar+erM+v["s"]*erc).hi);wu=(wr+erW).hi
    assert np.min(mu-ml)>=0 and np.min(zu-zl)>=0 and np.min(au-al)>=0
    return dict(ml=ml,mu=mu,zl=zl,zu=zu,al=al,au=au,wu=wu,error=radii,
                residual=z["residual"],growth_rate=z["growth_rate"])
