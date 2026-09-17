"""Read-only, evidence-bound system study summaries for the native UI."""
from pathlib import Path
import json
import math
from .io import sha256_file

SCHEMA = 'system_comparison_v1'


def load_review(root):
    root = Path(root)
    pointer = root/'config/evaluation/system_review.json'
    if not pointer.exists():
        return None
    selection = json.loads(pointer.read_text(encoding='utf-8'))
    path = Path(selection['report'])
    if not path.is_absolute(): path = root/path
    if sha256_file(path) != selection['sha256']:
        raise ValueError('Study report changed. Rebuild and explicitly register its evidence.')
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('schema') != SCHEMA:
        raise ValueError('Unsupported system comparison report')
    models = report['models']; ids = [m['id'] for m in models]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Study model identities must be unique')
    for path, expected in report['source_hashes'].items():
        if sha256_file(path) != expected:
            raise ValueError('Study evidence changed: '+str(path))
    return report


def eligible_takes(dataset, model_ids, selection='excluded'):
    """Exclude ancestry training for *every* model in a paired comparison."""
    result=[]
    for take, role in dataset['roles'].items():
        rows=[dataset['models'][mid].get(take) for mid in model_ids]
        if not all(rows): continue
        if selection == 'excluded' and any(r['data_use'] != 'Excluded from this model fitting' for r in rows): continue
        if selection == 'validation' and role != 'validation': continue
        result.append(take)
    return result


def equal_take_mean(values):
    valid=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(valid)/len(valid) if valid else None
