"""Independent audit of the final33-dose-cell adaptive model_2 finite cover.

Accepted sources: all rows of blocks00--06; firstthree dose rows of block07;
and both refined rows of refine07last. The original last row is explicitly
superseded, not silently accepted. No certificate trajectory is recomputed.
"""
from pathlib import Path
import argparse,json
import numpy as np
from audit_full_cover import (sha,delivery_bracket,poly_header,EXPECTED_FIT,
 EXPECTED_CONFIG,EXPECTED_PRIMITIVES,EXPECTED_MECHANISM)
HERE=Path(__file__).resolve().parent
DEFAULT=HERE.parent/'full32x16'


def audit(root,allow_incomplete=False):
    specs=[(f'block{i:02d}',i,4,list(range(4))) for i in range(7)]
    specs += [('block07',7,4,[0,1,2]),('refine07last','refine07last',2,[0,1])]
    rows=[];missing=[];failures=[];accepted_nodes=[];ref_ts=None;ref_es=None
    original_last=None;replacement=None;superseded={};first_fit=None;first_config=None
    for dirname,ident,nd,selection in specs:
        folder=root/dirname;path=folder/'block_result.json'
        if not path.exists():missing.append(str(path));continue
        try:
            r=json.loads(path.read_text());assert 'error' not in r
            assert r['block_index']==ident
            assert r['dose_cells']==nd and r['local_dose_cells']==nd and r['loading_cells']==16
            assert r['parameter_degrees']==[3,2] and r['nominal_full_cover_target_ng']==[20,100]
            assert r['time_cells']==1220 and r['bridge_end_hours']==96.
            assert r['all_positive_horizons_through48'] is True
            assert r['max_first_cell_production_upper']<0 and r['max_interior_cap']<0
            assert r['tail_M_matrix_margin']>0
            if dirname!='block07':
                assert r['passed'] is True and r['all_later_rectangles_strict'] is True
                assert r['upper_all_T_ge48']<0
            else:
                # This source is acceptable only through a genuinely verified subset.
                assert r['passed'] is False and r['all_later_rectangles_strict'] is False
            for name,digest in r['source_sha256'].items():assert sha(folder/name)==digest,(dirname,'source hash mismatch',name)
            for name,digest in [('committed_fit.json',EXPECTED_FIT),('model_config.json',EXPECTED_CONFIG),('certify.py',EXPECTED_PRIMITIVES),('repression_mechanisms.py',EXPECTED_MECHANISM)]:assert sha(folder/name)==digest
            if first_fit is None:first_fit=folder/'committed_fit.json';first_config=folder/'model_config.json'
            prop=folder/f'model_2_cubicquad_proposals_reference_{nd}_16_96.npz'
            with np.load(prop) as z:
                ks,es,ts=[z[k] for k in ['ks','es','ts']];shape,fortran,dtype=poly_header(z)
            assert shape==(nd,16,1220,3,4,3,8) and dtype==np.dtype('float64') and not fortran
            assert ks.shape==(nd+1,) and es.shape==(17,) and ts.shape==(1221,)
            assert np.isfinite(ks).all() and np.all(np.diff(ks)>0) and ks[0]>0
            assert np.array_equal(ks[[0,-1]],np.asarray(r['actual_delivered_interval']))
            assert np.isfinite(es).all() and np.all(np.diff(es)>0) and es[0]==0. and es[-1]==1.
            assert np.isfinite(ts).all() and np.all(np.diff(ts)>0) and ts[0]==0. and ts[-1]==96.
            assert 24. in ts and 48. in ts and ts[1]==0.0001
            if ref_ts is None:ref_ts=ts.copy();ref_es=es.copy()
            else:assert np.array_equal(ts,ref_ts) and np.array_equal(es,ref_es)
            if dirname=='block07':original_last=(float(ks[-2]),float(ks[-1]))
            if dirname=='refine07last':replacement=(float(ks[0]),float(ks[-1]))
            selected_nodes=ks[np.r_[selection,selection[-1]+1]]
            # Selections must be consecutive: no hidden skipped dose rows.
            assert selection==list(range(selection[0],selection[-1]+1))
            if accepted_nodes:assert accepted_nodes[-1]==selected_nodes[0],(dirname,'gap at segment boundary')
            accepted_nodes.extend(selected_nodes.tolist() if not accepted_nodes else selected_nodes[1:].tolist())
            cellpath=folder/'model_2_certificate_cells_reference.npz'
            with np.load(cellpath) as z:
                assert np.array_equal(z['ks'],ks) and np.array_equal(z['es'],es)
                values={n:np.asarray(z[n]) for n in ['upper24','upper48','later','all_time_ok']}
                for name,array in values.items():assert array.shape==(nd,16)
                assert values['all_time_ok'].dtype.kind=='b' and values['all_time_ok'][selection].all()
                for name in ['upper24','upper48','later']:
                    array=values[name];assert np.isfinite(array).all() and np.all(array[selection]<0),(dirname,name,'accepted subset not strict')
                    declared={'upper24':r['upper_cumulative_log_burden_slope']['24'],
                      'upper48':r['upper_cumulative_log_burden_slope']['48'],'later':r['upper_all_T_ge48']}[name]
                    assert float(array.max())==declared,(dirname,name,'record/array maximum differs')
                for name in ['terminal_absolute','bridge','tail']:
                    array=np.asarray(z[name]);assert np.isfinite(array).all() and np.all(array>=0),(dirname,name)
                masked=values['later'][selection];subind=tuple(map(int,np.unravel_index(masked.argmax(),masked.shape)))
                local=(selection[subind[0]],subind[1]);later=float(values['later'][local])
                limiting={'block':ident,'directory':dirname,'local_dose_cell':local[0],'loading_cell':local[1],
                    'delivered_interval':[float(ks[local[0]]),float(ks[local[0]+1])],
                    'loading_interval':[float(es[local[1]]),float(es[local[1]+1])],
                    'later_upper':later,'upper48_kernel':float(values['upper48'][local]),
                    'bridge_integral_upper':float(z['bridge'].reshape(nd,16)[local]),
                    'tail_integral_upper':float(z['tail'].reshape(nd,16)[local]),
                    'paired_S48_Q48_available':'S48' in z.files and 'Q48' in z.files}
                if limiting['paired_S48_Q48_available']:
                    limiting['S48_upper']=float(z['S48'].reshape(nd,16)[local]);limiting['Q48_upper']=float(z['Q48'].reshape(nd,16)[local]);limiting['adverse_budget_from_paired_difference']=later-limiting['S48_upper']
                if dirname=='block07':
                    last=values['later'][3]
                    superseded={'directory':dirname,'superseded_original_dose_rows':[3],
                      'superseded_rectangle_count':16,'delivered_interval':list(original_last),
                      'source_record_passed':r['passed'],'original_max_later_upper':float(last.max()),
                      'original_nonnegative_later_rectangle_count':int(np.count_nonzero(last>=0)),
                      'reason':'The original final dose row is replaced completely by two narrower verified dose cells; no failed cell is accepted.'}
                # State tubes were already validated by the verifier; inspect their archived outputs too.
                envpath=folder/f'validated_env_reference_{nd}_16_96.npz'
                with np.load(envpath) as env:
                    for low,high in [('ml','mu'),('zl','zu'),('al','au')]:
                        lowarr,hiarr=env[low],env[high]
                        assert lowarr.shape==(nd*16,1220) and hiarr.shape==lowarr.shape
                        assert np.isfinite(lowarr).all() and np.isfinite(hiarr).all() and np.all(lowarr>=0) and np.all(hiarr>=lowarr)
                    for name in ['error','residual']:
                        array=env[name];assert array.shape==(nd*16,1220,3) and np.isfinite(array).all() and np.all(array>=0)
                capture_dir=root/'adaptive_paired_reports'/dirname
                capture_path=capture_dir/'model_2_certificate_cells_reference.npz'
                capture_record=capture_dir/'paired_report.json'
                if capture_path.exists() and capture_record.exists():
                    cr=json.loads(capture_record.read_text())
                    assert cr['source_hashes_verified'] is True and cr['all_original_arrays_bitwise_identical'] is True
                    assert all(cr['array_checks'].values()) and cr['validated_env_sha256']==sha(envpath)
                    assert tuple(cr['limiting_later_cell_index'])==local
                    assert cr['accepted_original_dose_rows']==selection
                    with np.load(capture_path) as captured:
                        for name in z.files:assert np.array_equal(z[name],captured[name]),(dirname,'capture changed original array',name)
                        for name in ['S48','Q48']:
                            array=captured[name];assert array.shape==(nd,16) and np.isfinite(array).all()
                        assert float(captured['S48'][local])==cr['S48_upper']
                        assert float(captured['Q48'][local])==cr['Q48_upper']
                    assert cr['adverse_budget_upper']>=0 and np.isfinite(cr['adverse_budget_upper'])
                    assert cr['original_all_later_upper']==later and cr['paired_later_upper_recomputed']<0
                    assert abs(cr['paired_later_upper_recomputed']-later)<=8*np.spacing(max(1,abs(later)))
                    limiting.update(paired_S48_Q48_available=True,S48_upper=cr['S48_upper'],Q48_upper=cr['Q48_upper'],
                        adverse_budget_upper=cr['adverse_budget_upper'],paired_later_upper_recomputed=cr['paired_later_upper_recomputed'],
                        paired_capture_report_sha256=sha(capture_record),paired_capture_arrays_sha256=sha(capture_path),
                        paired_capture_scope='Comparison-only reporting replay; every original proof array matches bitwise.')
                rows.append({'block':ident,'directory':dirname,'source_record_passed':r['passed'],
                    'accepted_dose_rows':selection,'accepted_dose_cell_count':len(selection),'accepted_rectangle_count':16*len(selection),
                    'delivered_interval':[float(selected_nodes[0]),float(selected_nodes[-1])],
                    'upper24':float(values['upper24'][selection].max()),'upper48':float(values['upper48'][selection].max()),
                    'upper_all_T_ge48':float(values['later'][selection].max()),
                    'first_cell_production_upper':r['max_first_cell_production_upper'],
                    'interior_cap_upper':r['max_interior_cap'],'tail_M_matrix_margin':r['tail_M_matrix_margin'],
                    'limiting_later_rectangle':limiting,'source_sha256':r['source_sha256'],
                    'result_sha256':sha(path),'proposal_sha256':sha(prop),'cell_outputs_sha256':sha(cellpath),'state_envelopes_sha256':sha(envpath)})
        except Exception as exc:failures.append({'directory':dirname,'error':repr(exc)})
    complete=len(rows)==9 and not missing and not failures
    mapping={};global_bounds={}
    if complete:
        assert original_last==replacement,('replacement does not match original interval',original_last,replacement)
        nodes=np.asarray(accepted_nodes);assert nodes.shape==(34,) and np.all(np.diff(nodes)>0)
        assert sum(r['accepted_dose_cell_count'] for r in rows)==33
        assert sum(r['accepted_rectangle_count'] for r in rows)==528
        exact20,lo100,hi100,power=delivery_bracket(first_fit,first_config)
        from decimal import Decimal
        assert Decimal.from_float(float(nodes[0]))<=exact20
        assert Decimal.from_float(float(nodes[-1]))>=hi100
        mapping={'nominal_dose_ng':[20,100],'delivered_lower_float':float(nodes[0]),'delivered_upper_float':float(nodes[-1]),
          'exact_decimal_delivery_at20':str(exact20),'decimal_delivery_at100_lower':str(lo100),'decimal_delivery_at100_upper':str(hi100),
          'p_del':str(power),'strictly_increasing_delivery_map':True,
          'log_dose_slope_to_nominal_slope':'partial_q C=(p_del/q) partial_log(k0) C; p_del/q>0 on20<=q<=100',
          'initial_global_dose_cells':32,'global_cell_count':33,'loading_cells_each':16,'rectangle_count':528,
          'accepted_global_delivered_nodes':nodes.tolist(),'block_endpoints_exactly_contiguous':True,
          'refined_interval_exactly_matches_superseded_interval':True,'every_accepted_cell_output_strictly_negative':True}
        for name in ['upper24','upper48','upper_all_T_ge48','first_cell_production_upper','interior_cap_upper']:
            limiting=max(rows,key=lambda r:r[name]);global_bounds[name]={'upper':limiting[name],'limiting_block':limiting['block'],'directory':limiting['directory']}
        global_bounds['limiting_later_rectangle']=max((r['limiting_later_rectangle'] for r in rows),key=lambda r:r['later_upper'])
        global_bounds['minimum_tail_M_matrix_margin']=min(r['tail_M_matrix_margin'] for r in rows)
    report={'passed':complete,'complete':complete,'mapping':mapping,'global_bounds':global_bounds,'rows':rows,
        'missing':missing,'failures':failures,'superseded_original_failure':superseded,
        'scope':'Independent adaptive finite-cover aggregation/source/record audit; every accepted rectangle already has a residual/tube/sensitivity proof under the stated custom IEEE754 interval assumptions. No sampled sign or failed rectangle is promoted.'}
    (HERE/'adaptive_cover_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':complete,'accepted_sources':len(rows),'accepted_rectangles':sum(r['accepted_rectangle_count'] for r in rows),
        'missing':missing,'failures':failures,'mapping':mapping,'global_bounds':global_bounds},indent=2))
    if not complete and not allow_incomplete:raise SystemExit(1)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();audit(a.root.resolve(),a.allow_incomplete)
