"""Explicit scope for a single closed-loop response-gain update."""
SCHEMA='whip_response_gain_v1'


def validate(contract):
    if (contract.get('schema')!=SCHEMA or contract.get('parameter')!='feedforward_xy'
            or contract.get('bounds_scale')!=[.5,1.25]
            or contract.get('position_scale_m')!=.02 or contract.get('orientation_scale_rad')!=.05
            or contract.get('clock_check_offsets_s')!=[-.02,.02]
            or contract.get('regularization')!=.1
            or not contract.get('reason','').strip() or not contract.get('reviewed_by','').strip()
            or not contract.get('diagnostic_hashes')):
        raise ValueError('Reviewed single feedforward_xy response contract required')
    return contract
