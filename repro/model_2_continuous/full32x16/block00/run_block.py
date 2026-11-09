from pathlib import Path
import json,time,numpy as np,hashlib
import certify_model_2 as c
full=c.delivery_cover('reference');global_ks=np.geomspace(*full,33);global_ks[0],global_ks[-1]=full
interval=(float(global_ks[0]),float(global_ks[4]))
c.delivery_cover=lambda unused:interval
started=time.time()
try:r=c.run('reference',4,16,96.)
except Exception as e:
 import traceback;traceback.print_exc();r={'error':repr(e),'passed':False}
r['block_index']=0;r['actual_delivered_interval']=interval;r['nominal_full_cover_target_ng']=[20,100];r['global_dose_cells']=32;r['local_dose_cells']=4;r['parameter_degrees']=[3,2];r['total_seconds']=time.time()-started
r['source_sha256']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['certify.py','certify_model_2.py','model_2_rectangle_tubes.py','repression_mechanisms.py','committed_fit.json','model_config.json','run_block.py']}
Path('block_result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
