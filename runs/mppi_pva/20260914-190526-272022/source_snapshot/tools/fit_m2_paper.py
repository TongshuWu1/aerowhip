"""Prepare and fit the clean M1 batch with explicit residual budgets."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json
from experimental_data import whip_adaptation as adaptation
from experimental_data import whip_full_data as data
from experimental_data.whip_bounded_fit import fit


def main():
    batch=ROOT/'data/flight_batches/M1_paper_local_correction_20260913'
    comparison=ROOT/'runs/data_review/M1-paper-20260913'
    review=read_json(comparison/'review.json')
    review['cable_history_missingness']=dict(method='masked_regression_v1',minimum_observation_fraction=.8,
        authorization='User requested retaining valid observations despite occluded markers; same reviewed rule as M1 fitting.',
        details='Valid observations only; unchanged one-second causal history; no invented missing positions.')
    atomic_json(comparison/'review.masked_history.json',review)
    inputs=ROOT/'runs/adaptation/M2-paper-20260913-observed-inputs'
    adaptation.prepare(ROOT,batch,comparison,comparison/'review.masked_history.json',inputs,full_model=True)
    contract=data.default_contract()
    contract.update(candidate_id='M2',parent_id='M1',
        prior_whip_sources=[str(ROOT/'runs/adaptation/M1-paper-20260913-observed-inputs')],
        stage_budgets=dict(drone_residual=dict(maximum_updates=600,maximum_seconds=600),
            cable_residual=dict(maximum_updates=200,maximum_seconds=600)),
        stopping_authorization='User requested bounded residual fitting before the next M2 command correction; no changes to physical model or fitting objectives.')
    job=ROOT/'runs/adaptation/M2-paper-20260913'
    data.prepare(job,inputs,ROOT/'runs/adaptation/20260909-preliminary1-M0-v2',contract)
    fit(job)


if __name__=='__main__':main()
