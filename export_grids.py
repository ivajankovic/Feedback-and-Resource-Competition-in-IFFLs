"""Export the accepted parameter rectangles and time nodes after a full replay.

Nominal inverse-map values are approximate labels; delivered-burden float
endpoints and the validated exact delivery bracketing define the proof cover.
"""
from pathlib import Path
import numpy as np,csv,json
p=Path(__file__).resolve().parent
rows=[]
fit=json.loads((p/'repro/committed_fit.json').read_text())
cfg=json.loads((p/'repro/model_config.json').read_text())
def nominal(k,mech):
 arm=fit['parameters']['nuisance_per_arm']['reference_'+mech+'_RNA']
 K=cfg['initial_conditions_RNA']['RNA_Burden']*arm['kappa_burden_circuit']
 return 20.*(k/K)**(1./arm['p_dose_circuit'])
with np.load(p/'repro/certificate_cells_reference.npz') as z,np.load(p/'repro/tail_certificate_cells_reference.npz') as tail:
 for i in range(32):
  for j in range(8):
   rows.append(['model_1','repro',i,j,z['ks'][i],z['ks'][i+1],nominal(z['ks'][i],'model_1'),nominal(z['ks'][i+1],'model_1'),z['es'][j],z['es'][j+1],48,96,.002,z['upper24'][i,j],z['upper48'][i,j],tail['upper'][i,j]])
for unit in [f'block{i:02d}' for i in range(8)]+['refine07last']:
 folder='repro/model_2_continuous/full32x16/'+unit
 with np.load(p/folder/'model_2_certificate_cells_reference.npz') as z:
  nd=3 if unit=='block07' else len(z['ks'])-1
  for i in range(nd):
   for j in range(16):
    rows.append(['model_2',folder,i,j,z['ks'][i],z['ks'][i+1],nominal(z['ks'][i],'model_2'),nominal(z['ks'][i+1],'model_2'),z['es'][j],z['es'][j+1],48,96,.0001,z['upper24'][i,j],z['upper48'][i,j],z['later'][i,j]])
with (p/'certificate_rectangles.csv').open('w',newline='') as out:
 writer=csv.writer(out)
 writer.writerow(['model','proof_source_folder','local_dose_cell_zero_based','loading_cell_zero_based','delivered_burden_lower_C','delivered_burden_upper_C','nominal_dose_lower_approx_ng','nominal_dose_upper_approx_ng','loading_lower','loading_upper','T0_h','T1_h','first_production_sign_interval_h','upper_S24_logburden','upper_S48_kernel_logburden','upper_all_T_ge48_logburden'])
 writer.writerows(rows)
with (p/'certificate_time_nodes.csv').open('w',newline='') as out:
 writer=csv.writer(out);writer.writerow(['model','node_index_zero_based','time_h','phase'])
 for mech,path in [('model_1','repro/corner_polynomials_reference.npz'),('model_2','repro/model_2_continuous/full32x16/block00/model_2_cubicquad_proposals_reference_4_16_96.npz')]:
  # Full replay regenerates these untrusted proposal files; only their time nodes are exported.
  f=p/path
  with np.load(f) as z:ts=z['ts'].copy()
  if mech=='model_1':ts=np.r_[ts,np.linspace(48.,96.,481)[1:]]
  for i,t in enumerate(ts):writer.writerow([mech,i,t,'finite' if t<=48 else 'bridge'])
print({'rectangles':len(rows),'model_1':sum(x[0]=='model_1' for x in rows),'model_2':sum(x[0]=='model_2' for x in rows)})
