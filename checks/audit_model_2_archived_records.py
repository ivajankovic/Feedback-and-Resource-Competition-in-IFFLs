"""Audit the distributed model_2 proof records without regenerating omitted caches.

This is an archived-record audit, not fresh interval residual validation.
Run from the package root, or supply --root /absolute/package/root.
"""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import argparse
import hashlib
import json
import math
import numpy as np


EXPECTED = {
    'committed_fit.json': '119640ce00c1bfbb28f74964d5128a4a12a7f4fccdab3679ef9dd8646223b517',
    'model_config.json': 'abfe9d9491c5c2fd4b91530d468ab28bd3182b9281600b9f7b4aa8e85661f6f0',
    'certify.py': '974c27eeccd009a48b5e8cbec0dc2284c6028824f411bca68e70b648ea581930',
    'repression_mechanisms.py': '800ad985cb38b375c5b7fe2223e476d2c6b79160dd6e26319cc1b62373495042',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delivery_bracket(folder):
    fit = json.loads((folder/'committed_fit.json').read_text(), parse_float=Decimal, parse_int=Decimal)
    cfg = json.loads((folder/'model_config.json').read_text(), parse_float=Decimal, parse_int=Decimal)
    arm = fit['parameters']['nuisance_per_arm']['reference_model_2_RNA']
    with localcontext() as ctx:
        ctx.prec = 200
        ctx.rounding = ROUND_HALF_EVEN
        k20 = cfg['initial_conditions_RNA']['RNA_Burden'] * arm['kappa_burden_circuit']
        power = arm['p_dose_circuit']
        assert k20 > 0 and power > 0
        log5 = Decimal(5).ln()
        lo = (k20 * (log5.next_minus()*power).next_minus().exp().next_minus()).next_minus()
        hi = (k20 * (log5.next_plus()*power).next_plus().exp().next_plus()).next_plus()
        return k20, lo, hi, power


def audit(root):
    base = root/'repro/model_2_continuous/full32x16'
    specs = [(f'block{i:02d}', list(range(4))) for i in range(7)]
    specs += [('block07', [0, 1, 2]), ('refine07last', [0, 1])]
    nodes, rows, failures = [], [], []
    reference_loading = None
    original_last = replacement = None
    for name, selection in specs:
        try:
            folder = base/name
            record_path = folder/'block_result.json'
            record = json.loads(record_path.read_text())
            nd = 2 if name == 'refine07last' else 4
            assert record['dose_cells'] == record['local_dose_cells'] == nd
            assert record['loading_cells'] == 16
            assert record['parameter_degrees'] == [3, 2]
            assert record['time_cells'] == 1220 and record['bridge_end_hours'] == 96.
            assert record['nominal_full_cover_target_ng'] == [20, 100]
            assert record['all_positive_horizons_through48'] is True
            assert record['max_first_cell_production_upper'] < 0
            assert record['max_interior_cap'] < 0
            assert record['tail_M_matrix_margin'] > 0
            assert record['passed'] is (name != 'block07')
            assert record['all_later_rectangles_strict'] is (name != 'block07')
            for filename, expected in record['source_sha256'].items():
                assert sha(folder/filename) == expected, ('source hash', filename)
            for filename, expected in EXPECTED.items():
                assert sha(folder/filename) == expected, ('fixed input or primitive hash', filename)
            cell_path = folder/'model_2_certificate_cells_reference.npz'
            with np.load(cell_path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            ks, es = arrays['ks'], arrays['es']
            assert ks.shape == (nd+1,) and es.shape == (17,)
            assert np.isfinite(ks).all() and np.all(np.diff(ks) > 0)
            assert np.isfinite(es).all() and np.all(np.diff(es) > 0) and es[0] == 0. and es[-1] == 1.
            assert np.array_equal(ks[[0, -1]], np.asarray(record['actual_delivered_interval']))
            if reference_loading is None: reference_loading = es.copy()
            else: assert np.array_equal(es, reference_loading)
            assert selection == list(range(selection[0], selection[-1]+1))
            selected_nodes = ks[np.r_[selection, selection[-1]+1]]
            if nodes: assert nodes[-1] == selected_nodes[0], 'gap or overlapping accepted cells'
            nodes.extend(selected_nodes.tolist() if not nodes else selected_nodes[1:].tolist())
            for key in ['upper24', 'upper48', 'later']:
                values = arrays[key]
                assert values.shape == (nd, 16) and np.isfinite(values).all()
                assert np.all(values[selection] < 0), ('nonnegative accepted slope', key)
                declared = {'upper24': record['upper_cumulative_log_burden_slope']['24'],
                            'upper48': record['upper_cumulative_log_burden_slope']['48'],
                            'later': record['upper_all_T_ge48']}[key]
                assert float(values.max()) == declared, ('array/record maximum', key)
            assert arrays['all_time_ok'].shape == (nd, 16)
            assert arrays['all_time_ok'].dtype.kind == 'b' and arrays['all_time_ok'][selection].all()
            for key in ['terminal_absolute', 'bridge', 'tail']:
                assert np.isfinite(arrays[key]).all() and np.all(arrays[key] >= 0), key
            if name == 'block07':
                original_last = ks[-2:].tolist()
                superseded = {'delivered_interval': original_last,
                              'nonnegative_later_rectangles': int(np.count_nonzero(arrays['later'][3] >= 0)),
                              'worst_later_upper': float(arrays['later'][3].max())}
                assert superseded['nonnegative_later_rectangles'] == 5
                assert superseded['worst_later_upper'] > 0
            if name == 'refine07last': replacement = ks[[0, -1]].tolist()
            capture = base/'adaptive_paired_reports'/name
            paired = json.loads((capture/'paired_report.json').read_text())
            assert paired['source_hashes_verified'] is True
            assert paired['all_original_arrays_bitwise_identical'] is True
            assert all(paired['array_checks'].values())
            assert paired['accepted_original_dose_rows'] == selection
            with np.load(capture/'model_2_certificate_cells_reference.npz', allow_pickle=False) as data:
                for key, value in arrays.items():
                    assert np.array_equal(value, data[key]), ('reporting capture changed original array', key)
                loc = tuple(paired['limiting_later_cell_index'])
                assert loc[0] in selection and 0 <= loc[1] < 16
                assert float(data['S48'][loc]) == paired['S48_upper']
                assert float(data['Q48'][loc]) == paired['Q48_upper']
            assert paired['adverse_budget_upper'] >= 0
            actual_sum = paired['S48_upper'] + paired['adverse_budget_upper']
            assert actual_sum < 0 and paired['paired_later_upper_recomputed'] < 0
            assert paired['original_all_later_upper'] == float(arrays['later'][loc])
            assert abs(actual_sum-paired['original_all_later_upper']) <= 8*np.spacing(max(1, abs(actual_sum)))
            rows.append({'unit': name, 'accepted_dose_rows': selection,
                         'accepted_rectangles': 16*len(selection),
                         'upper24': float(arrays['upper24'][selection].max()),
                         'upper48': float(arrays['upper48'][selection].max()),
                         'later_upper': float(arrays['later'][selection].max()),
                         'first_cell_production_upper': record['max_first_cell_production_upper'],
                         'interior_cap_upper': record['max_interior_cap'],
                         'tail_M_matrix_margin': record['tail_M_matrix_margin'],
                         'recorded_execution_seconds': record['total_seconds'],
                         'result_sha256': sha(record_path), 'cell_outputs_sha256': sha(cell_path),
                         'pair': {key: paired[key] for key in ['S48_upper', 'Q48_upper', 'adverse_budget_upper', 'paired_later_upper_recomputed']}})
        except Exception as error:
            failures.append({'unit': name, 'error': repr(error)})
    complete = len(rows) == 9 and not failures
    cover, bounds = {}, {}
    if complete:
        assert original_last == replacement
        assert len(nodes) == 34 and np.all(np.diff(nodes) > 0)
        assert sum(row['accepted_rectangles'] for row in rows) == 528
        k20, lo100, hi100, power = delivery_bracket(base/'block00')
        assert Decimal.from_float(nodes[0]) <= k20
        assert Decimal.from_float(nodes[-1]) >= hi100
        cover = {'dose_ng': [20, 100], 'loading': [0, 1], 'dose_cells': 33,
                 'loading_cells': 16, 'rectangles': 528, 'delivered_nodes': nodes,
                 'lower_endpoint_covers_exact_delivery': True,
                 'upper_endpoint_covers_exact_delivery': True,
                 'positive_delivery_exponent': str(power),
                 'superseded_final_row': superseded,
                 'replacement_has_identical_combined_endpoints': True}
        for key in ['upper24', 'upper48', 'later_upper', 'first_cell_production_upper', 'interior_cap_upper']:
            bounds[key] = max(row[key] for row in rows)
        bounds['minimum_tail_M_matrix_margin'] = min(row['tail_M_matrix_margin'] for row in rows)
    return {'passed': complete, 'scope': 'Archived-record audit; not fresh interval residual validation.',
            'checks_performed': ['current recorded source/input SHA256 identities',
                                 'accepted per-rectangle NPZ slope bounds and horizon-gate records',
                                 'contiguous adaptive delivery cover and exact endpoint bracketing',
                                 'whole loading interval and positive nominal-dose conversion',
                                 'explicit exclusion of failed original last row',
                                 'paired accumulated-margin/tail-budget arithmetic',
                                 'bitwise identity of paired reporting/original output arrays'],
            'not_reexecuted': ['omitted polynomial proposals and state-enclosure caches',
                               'polynomial residual/state-tube validation',
                               'finite-time flow and analytic-tail certificate execution',
                               'formal verification of numerical software or hardware'],
            'cover': cover, 'global_archived_bounds': bounds, 'units': rows, 'failures': failures,
            'sum_recorded_execution_seconds': sum(row['recorded_execution_seconds'] for row in rows),
            'runtime_note': 'Recorded nine-unit sum, not elapsed runtime of this archived-record audit or total study elapsed time.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('audit_model_2_archived_records.json'))
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'scope', 'global_archived_bounds', 'failures']}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
