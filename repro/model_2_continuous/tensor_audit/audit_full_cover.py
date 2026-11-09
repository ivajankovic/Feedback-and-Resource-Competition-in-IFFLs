"""Audit coverage and every recorded certificate gate for the full model_2 run.

This does not rerun polynomial residual validation. It verifies the exact
parameter/time cover, current source/input identities, recorded all-horizon
acceptance gates, and the per-rectangle output arrays. Use only after the
unchanged verifier has produced all eight block records.
"""
from pathlib import Path
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
import argparse,json,hashlib,sys
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT=HERE.parent/'full32x16'
EXPECTED_FIT='119640ce00c1bfbb28f74964d5128a4a12a7f4fccdab3679ef9dd8646223b517'
EXPECTED_CONFIG='abfe9d9491c5c2fd4b91530d468ab28bd3182b9281600b9f7b4aa8e85661f6f0'
EXPECTED_PRIMITIVES='974c27eeccd009a48b5e8cbec0dc2284c6028824f411bca68e70b648ea581930'
EXPECTED_MECHANISM='800ad985cb38b375c5b7fe2223e476d2c6b79160dd6e26319cc1b62373495042'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def delivery_bracket(fit,config):
    f=json.loads(fit.read_text(),parse_float=Decimal,parse_int=Decimal)
    b=json.loads(config.read_text(),parse_float=Decimal,parse_int=Decimal)
    nu=f['parameters']['nuisance_per_arm']['reference_model_2_RNA']
    with localcontext() as ctx:
        ctx.prec=200;ctx.rounding=ROUND_HALF_EVEN
        rho=nu['kappa_burden_circuit'];power=nu['p_dose_circuit'];base=b['initial_conditions_RNA']['RNA_Burden']
        assert rho>0 and power>0 and base>0
        exact20=base*rho
        log5=Decimal(5).ln()
        p_lo=(log5.next_minus()*power).next_minus()
        p_hi=(log5.next_plus()*power).next_plus()
        e_lo=p_lo.exp().next_minus();e_hi=p_hi.exp().next_plus()
        low100=(exact20*e_lo).next_minus();high100=(exact20*e_hi).next_plus()
        return exact20,low100,high100,power

def poly_header(npz):
    with npz.zip.open('poly.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise AssertionError(('unsupported npy header',version))
    return shape,fortran,np.dtype(dtype)

def check(root,allow_incomplete=False):
    rows=[];missing=[];failures=[];bounds={};nodes_by_block={};reference_ts=None;reference_es=None
    source_hashes={};actual_fit=None;actual_config=None
    for i in range(8):
        block=root/f'block{i:02d}'
        record=block/'block_result.json'
        if not record.exists():missing.append(str(record));continue
        try:
            result=json.loads(record.read_text())
            assert result['passed'] is True,(i,'verifier reported failure',result.get('error'))
            assert result['block_index']==i
            assert result['global_dose_cells']==32 and result['local_dose_cells']==4
            assert result['parameter_degrees']==[3,2]
            assert result['nominal_full_cover_target_ng']==[20,100]
            assert result['dose_cells']==4 and result['loading_cells']==16
            assert result['time_cells']==1220 and result['bridge_end_hours']==96.
            assert result['all_positive_horizons_through48'] is True
            assert result['all_later_rectangles_strict'] is True
            assert result['max_first_cell_production_upper']<0
            assert result['max_interior_cap']<0
            assert result['tail_M_matrix_margin']>0
            assert result['upper_all_T_ge48']<0
            assert result['upper_cumulative_log_burden_slope']['24']<0
            assert result['upper_cumulative_log_burden_slope']['48']<0
            for name,digest in result['source_sha256'].items():
                assert sha(block/name)==digest,(i,'source hash mismatch',name)
            assert sha(block/'committed_fit.json')==EXPECTED_FIT
            assert sha(block/'model_config.json')==EXPECTED_CONFIG
            assert sha(block/'certify.py')==EXPECTED_PRIMITIVES
            assert sha(block/'repression_mechanisms.py')==EXPECTED_MECHANISM
            if actual_fit is None:actual_fit=block/'committed_fit.json';actual_config=block/'model_config.json'
            prop=block/'model_2_cubicquad_proposals_reference_4_16_96.npz'
            with np.load(prop) as data:
                ks,es,ts=[data[k] for k in ['ks','es','ts']]
                shape,fortran,dtype=poly_header(data)
            assert shape==(4,16,1220,3,4,3,8) and dtype==np.dtype('float64')
            assert not fortran
            assert ks.shape==(5,) and es.shape==(17,) and ts.shape==(1221,)
            assert np.isfinite(ks).all() and np.all(np.diff(ks)>0) and ks[0]>0
            assert np.isfinite(es).all() and np.all(np.diff(es)>0)
            assert es[0]==0. and es[-1]==1.
            assert np.isfinite(ts).all() and np.all(np.diff(ts)>0)
            assert ts[0]==0. and ts[-1]==96. and 24. in ts and 48. in ts
            assert ts[1]==0.0001
            if reference_ts is None:reference_ts=ts;reference_es=es
            else:assert np.array_equal(reference_ts,ts) and np.array_equal(reference_es,es)
            declared=np.array(result['actual_delivered_interval'])
            assert np.array_equal(ks[[0,-1]],declared),(i,'declared interval mismatch')
            if i-1 in bounds:assert bounds[i-1][1]==ks[0],(i,'gap or overlapping block endpoints')
            bounds[i]=(float(ks[0]),float(ks[-1]));nodes_by_block[i]=ks.copy()
            with np.load(block/'model_2_certificate_cells_reference.npz') as data:
                assert np.array_equal(data['ks'],ks) and np.array_equal(data['es'],es)
                for name in ['upper24','upper48','later']:
                    vals=data[name]
                    assert vals.shape==(4,16) and np.isfinite(vals).all() and np.all(vals<0),(i,name)
                    key={'upper24':('upper_cumulative_log_burden_slope','24'),'upper48':('upper_cumulative_log_burden_slope','48'),'later':('upper_all_T_ge48',None)}[name]
                    declared_max=result[key[0]][key[1]] if key[1] else result[key[0]]
                    assert float(vals.max())==declared_max,(i,name,'max mismatch')
                assert data['all_time_ok'].shape==(4,16) and data['all_time_ok'].dtype.kind=='b' and data['all_time_ok'].all()
                for name in ['terminal_absolute','bridge','tail']:
                    vals=data[name];assert np.isfinite(vals).all() and np.all(vals>=0),(i,name)
            rows.append({'block':i,'delivered_interval':bounds[i],
                'upper24':result['upper_cumulative_log_burden_slope']['24'],
                'upper48':result['upper_cumulative_log_burden_slope']['48'],
                'upper_all_T_ge48':result['upper_all_T_ge48'],
                'first_cell_production_upper':result['max_first_cell_production_upper'],
                'interior_cap_upper':result['max_interior_cap'],
                'tail_M_matrix_margin':result['tail_M_matrix_margin'],
                'source_sha256':result['source_sha256'],
                'result_sha256':sha(record),'proposal_sha256':sha(prop),
                'cell_outputs_sha256':sha(block/'model_2_certificate_cells_reference.npz')})
        except Exception as e:
            failures.append({'block':i,'error':repr(e)})
    complete=len(rows)==8 and not missing and not failures
    mapping={}
    if complete:
        global_nodes=[]
        for i in range(8):
            if i:assert bounds[i-1][1]==bounds[i][0]
            global_nodes.extend(nodes_by_block[i].tolist() if i==0 else nodes_by_block[i][1:].tolist())
        nodes=np.array(global_nodes);assert nodes.shape==(33,) and np.all(np.diff(nodes)>0)
        exact20,lo100,hi100,power=delivery_bracket(actual_fit,actual_config)
        assert Decimal.from_float(float(nodes[0]))<=exact20
        assert Decimal.from_float(float(nodes[-1]))>=hi100
        mapping={'nominal_dose_ng':[20,100],'delivered_lower_float':float(nodes[0]),'delivered_upper_float':float(nodes[-1]),
          'exact_decimal_delivery_at20':str(exact20),'decimal_delivery_at100_lower':str(lo100),'decimal_delivery_at100_upper':str(hi100),
          'p_del':str(power),'strictly_increasing_delivery_map':True,
          'log_dose_slope_to_nominal_slope':'partial_q C=(p_del/q) partial_log(k0) C; factor positive for20<=q<=100',
          'global_cell_count':32,'loading_cells_each':16,'rectangle_count':512,
          'block_endpoints_exactly_contiguous':True,'every_cell_output_strictly_negative':True}
    global_bounds={}
    if complete:
        for key in ['upper24','upper48','upper_all_T_ge48','first_cell_production_upper','interior_cap_upper']:
            limiting=max(rows,key=lambda r:r[key]);global_bounds[key]={'upper':limiting[key],'limiting_block':limiting['block']}
        limiting=[]
        for row in rows:
            i=row['block'];block=root/f'block{i:02d}'
            with np.load(block/'model_2_certificate_cells_reference.npz') as data:
                index=tuple(map(int,np.unravel_index(np.argmax(data['later']),data['later'].shape)))
                later=float(data['later'][index]);d=nodes_by_block[i];eps=reference_es
                rec={'block':i,'local_dose_cell':index[0],'loading_cell':index[1],'global_dose_cell':4*i+index[0],
                     'delivered_interval':[float(d[index[0]]),float(d[index[0]+1])],
                     'loading_interval':[float(eps[index[1]]),float(eps[index[1]+1])],'later_upper':later,
                     'bridge_integral_upper':float(data['bridge'].reshape(4,16)[index]),
                     'tail_integral_upper':float(data['tail'].reshape(4,16)[index]),
                     'upper48_kernel':float(data['upper48'][index]),
                     'paired_S48_Q48_available':'S48' in data.files and 'Q48' in data.files}
                if rec['paired_S48_Q48_available']:
                    rec['S48_upper']=float(data['S48'].reshape(4,16)[index]);rec['Q48_upper']=float(data['Q48'].reshape(4,16)[index]);rec['adverse_budget_from_paired_difference']=later-rec['S48_upper']
                limiting.append(rec)
        global_bounds['limiting_later_rectangle']=max(limiting,key=lambda r:r['later_upper'])
        global_bounds['minimum_tail_M_matrix_margin']=min(r['tail_M_matrix_margin'] for r in rows)
    report={'passed':complete,'complete':complete,'global_bounds':global_bounds,'rows':rows,'missing':missing,'failures':failures,'mapping':mapping,
      'scope':'Independent cover/record audit; every block must already pass residual/tube/sensitivity validation under stated custom IEEE754 interval assumptions.'}
    (HERE/'full_cover_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':complete,'blocks_accepted':len(rows),'missing':missing,'failures':failures,'mapping':mapping,'global_bounds':global_bounds},indent=2))
    if not complete and not allow_incomplete:raise SystemExit(1)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();check(a.root.resolve(),a.allow_incomplete)
