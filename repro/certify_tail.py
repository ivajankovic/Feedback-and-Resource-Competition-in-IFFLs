"""Outward-rounded all-later-horizon extension of the 48-hour certificate.
Uses only certified terminal sensitivity bounds and analytical RNA envelopes.
"""
import json, math
from pathlib import Path
import numpy as np
import certify as c
ROOT=Path(__file__).parent
I=c.I

def run():
    p,v,fit,base=c.load_parameters();data=np.load(ROOT/'certified_terminal_reference.npz')
    ks=data['ks'];es=data['es'];nd=len(ks)-1;ne=len(es)-1
    nc=nd*ne
    for name in ['P','H','Qupper','Supper','all_time_ok']:
        if data[name].shape!=(nc,):
            raise AssertionError('terminal array shape does not match its parameter grid: '+name)
    for name in ['P','H','Qupper','Supper']:
        if not np.all(np.isfinite(data[name])):
            raise AssertionError('nonfinite terminal bound: '+name)
    P=data['P'].copy();H=data['H'].copy()
    if np.any(P<0) or np.any(H<0):
        raise AssertionError('nonnegative terminal comparison bounds required')
    klo=I(np.repeat(ks[:-1],ne));khi=I(np.repeat(ks[1:],ne))
    # The last polynomial cell gives a valid lower bound at its terminal point.
    props=np.load(ROOT/'corner_polynomials_reference.npz')
    if not (np.array_equal(ks,props['ks']) and np.array_equal(es,props['es'])):
        raise AssertionError('terminal and polynomial parameter grids must match exactly')
    cr,zr,E,rho,G=c.state_bounds(props,v)
    Z0=np.maximum(0,c.DOWN(zr.lo[1:,1:,-1]-E[1:,1:])).reshape(-1)
    M48=(v['M0']*c.exp_neg(v['beta']*48))
    c.DPAR=float(v['d'].lo)
    area=np.zeros(nc);t0=48.;t1=96.;step=.1
    nodes=np.linspace(t0,t1,481)
    for t,te in zip(nodes[:-1],nodes[1:]):
        dt=I(te)-I(t)
        ex=c.exp_neg(v['beta']*I(t));mu=(v['M0']*ex).hi
        # Integrate M(s)=M48 exp[-beta(s-48)] exactly in Z' >= -(d+eM)Z.
        lag=I(te)-I(t0)
        zlo=(I(Z0)*c.exp_neg(v['d']*lag+v['e']*M48/v['beta']*(1-c.exp_neg(v['beta']*lag)))).lo
        a=(v['h']+v['e']*I(zlo)).lo+np.zeros(nc)
        b=(v['e']*I(mu)).hi+np.zeros(nc)
        F=(v['gR']*v['R']*v['r0']/v['kR']*ex).hi
        cc=(v['d']+I(F)/v['kG']).hi+np.zeros(nc)
        f=(I(F)*khi/v['kG']*ex).hi
        gam=np.maximum(np.maximum((I(b)-I(a)).hi,(I(cc)-I(c.DPAR)).hi),0)
        Hsup=(c.exp_pos(I(gam)*dt)*(I(np.maximum(P,H))+I(f)*dt)).hi
        Psup=np.maximum(P,(I(b)*I(Hsup)/I(a)).hi)
        # B is decreasing in time, so its minimum on the cell is at te.
        Bend=(1+(v['r0']/v['kR']+klo/v['kG'])*c.exp_neg(v['beta']*I(te))).lo
        area=(I(area)+dt*I(Psup)/I(Bend)).hi
        P,H=c.pos_exp_action(P,H,a,b,cc,f,dt)
    ex=c.exp_neg(v['beta']*I(t1))
    b=(v['e']*v['M0']*ex).hi+np.zeros(nc)
    F=(v['gR']*v['R']*v['r0']/v['kR']*ex).hi
    cc=(v['d']+I(F)/v['kG']).hi+np.zeros(nc)
    a=float(v['h'].lo);d=float(v['d'].lo)
    det=(I(a)*I(d)-I(b)*I(cc)).lo
    if not np.all(det>0):raise AssertionError('tail comparison not Hurwitz')
    f=(I(F)*khi/v['kG']*ex).hi
    farea=(I(f)/(2*v['beta'])).hi
    tail_area=((I(d)*I(P)+I(b)*(I(H)+I(farea)))/I(det)).hi
    extra=(I(np.maximum(data['Qupper'],0))/v['d']+v['vc']/v['d']*(I(area)+I(tail_area))).hi
    final=(I(data['Supper'])+I(extra)).hi
    result={'start_horizon':t0,'bridge_end':t1,'bridge_cells':len(nodes)-1,'maximum_endpoint_upper':float(data['Qupper'].max()),'max_positive_slope_tail_budget':float(extra.max()),'max_all_later_horizon_upper':float(final.max()),'all_cells_strictly_negative':bool(np.all(final<0)),'minimum_tail_matrix_determinant':float(det.min()),'cell_bounds':final.reshape(nd,ne).tolist(),'method':'positive sensitivity tail comparison; outward-rounded bridge and analytic stable-tail resolvent'}
    (ROOT/'tail_certificate_reference.json').write_text(json.dumps(result,indent=2));np.savez_compressed(ROOT/'tail_certificate_cells_reference.npz',ks=ks,es=es,upper=final.reshape(nd,ne),extra=extra.reshape(nd,ne))
    print(json.dumps({k:v for k,v in result.items() if k!='cell_bounds'}),flush=True)
    return result
if __name__=='__main__':run()
