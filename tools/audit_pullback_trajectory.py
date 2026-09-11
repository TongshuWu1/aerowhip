from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from matplotlib.figure import Figure
from simulator.workflow import read_json
from experimental_data.io import atomic_json
from deployment.whip_diagnostics import draw

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rehearsal',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.rehearsal/'rehearsal.npz') as z:arrays={k:z[k].copy() for k in z.files}
    figure=Figure(figsize=(10,9),layout='constrained')
    metrics=draw(figure,arrays,read_json(args.rehearsal/'rehearsal.json'),read_json(args.rehearsal/'settings.json'))
    figure.savefig(args.output/'pullback.png',dpi=180);atomic_json(args.output/'metrics.json',metrics);print(metrics)
