"""Read historical force-physics fixtures without importing a policy trainer."""
import json
from pathlib import Path

def load_configs(config_directory=None):
    directory=Path(config_directory) if config_directory else Path(__file__).resolve().parents[1]/'config'
    return tuple(json.loads((directory/name).read_text(encoding='utf-8'))
                 for name in ('model.json','task.json','ppo.json'))
