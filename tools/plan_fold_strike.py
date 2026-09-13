"""Plan a targeted cable strike offline, using an explicit model and templates."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Override the model in the saved profile')
    parser.add_argument('--templates', type=Path,
                        help='Frozen proposal_baselines.npz; command initialization only')
    parser.add_argument('--settings', type=Path, default=ROOT/'config/pva/systematic_strike.json')
    parser.add_argument('--name', default='spline-targeted-strike')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--check', action='store_true', help='Validate inputs without running an optimizer')
    args = parser.parse_args()
    from planning.pva_job import prepare, run, validate_settings
    cfg = json.loads(args.settings.read_text(encoding='utf-8'))
    model = args.model.resolve() if args.model else (ROOT/cfg['model_path']).resolve()
    from planning.strike_objective import uses_templates
    if args.templates and not uses_templates(cfg):raise ValueError('From-scratch mode does not accept saved templates')
    templates = args.templates.resolve() if args.templates else (ROOT/cfg.get(
        'proposal_templates_path','workspace/baseline/planner/proposal_baselines.npz')).resolve()
    cfg.update(model_path=str(model), device=args.device)
    validate_settings(cfg)
    if not model.is_file():
        raise FileNotFoundError(model)
    if not cfg.get('spline_seed_directory') and uses_templates(cfg) and not templates.is_file():
        raise FileNotFoundError(templates)
    if cfg.get('spline_seed_directory'):
        seed_root=ROOT/cfg['spline_seed_directory']
        for filename in ('initial_proposal.npz','proposal_baselines.npz',cfg['trajectory_objective']['reference_file']):
            if not (seed_root/filename).is_file():raise FileNotFoundError(seed_root/filename)
        templates=seed_root/'proposal_baselines.npz'
    if args.check:
        print(json.dumps(dict(model=str(model),templates=str(templates) if uses_templates(cfg) else None,
            objective=cfg['trajectory_objective']['schema'],optimizer_started=False),indent=2))
        return
    job, _ = prepare(ROOT, cfg, args.name,
        development_review='Explicit offline simulation planning; physical performance remains unvalidated.')
    if not cfg.get('spline_seed_directory') and uses_templates(cfg):shutil.copy2(templates, job/'proposal_baselines.npz')
    print(job, flush=True)
    run(job)


if __name__ == '__main__':
    main()
