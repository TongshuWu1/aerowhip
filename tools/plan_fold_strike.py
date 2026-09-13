"""Plan a travelling-fold strike offline, using an explicit model and templates."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True,
                        help='Frozen proposal_baselines.npz; command initialization only')
    parser.add_argument('--settings', type=Path, default=ROOT/'config/pva/systematic_strike.json')
    parser.add_argument('--name', default='travelling-fold-strike')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    from planning.pva_job import prepare, run
    cfg = json.loads(args.settings.read_text(encoding='utf-8'))
    cfg.update(model_path=str(args.model.resolve()), device=args.device)
    templates = args.templates.resolve()
    if not templates.is_file():
        raise FileNotFoundError(templates)
    job, _ = prepare(ROOT, cfg, args.name,
        development_review='Explicit offline simulation planning; physical performance remains unvalidated.')
    shutil.copy2(templates, job/'proposal_baselines.npz')
    print(job, flush=True)
    run(job)


if __name__ == '__main__':
    main()
