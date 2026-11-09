import json, math, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, simpson

ROOT=Path(__file__).parent
fit=json.loads((ROOT/'committed_fit.json').read_text())
raw=json.loads((ROOT/'model_config.json').read_text())
p=raw['params']|fit['parameters']['fitted_shared']
M0=raw['initial_conditions_RNA']['RNA_Output']
r0=raw['initial_conditions_RNA']['RNA_Repressor']
kbase=raw['initial_conditions_RNA']['RNA_Burden']
beta=p['beta']; h=beta+p['eta_minus_model_1']; d=p['delta']+p['mu']; e=p['eta_plus_model_1']
gR=p['gamma']/p['length_model_1']; gG=p['gamma']/p['length_output']
kG=(gG+p['kappa_minus'])/(p['kappa_plus']*p['length_output']/p['ribo_footprint_size'])
kR=(gR+p['kappa_minus'])/(p['kappa_plus']*p['length_model_1']/p['ribo_footprint_size'])
R=p['total_rtl']; vc=gG*R/kG

def inputs(t,k0):
    exp=math.exp(-beta*t)
    M=M0*exp; rr=r0/kR*exp; b=k0/kG*exp
    return M,1+rr+b,b,gR*R*rr

