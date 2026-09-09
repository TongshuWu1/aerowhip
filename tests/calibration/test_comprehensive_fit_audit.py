import json
from pathlib import Path
import numpy as np
import torch

from simulator.cable import CableConfiguration
from experimental_data.comprehensive_fit_audit import reconstruct,assemble


def payload():
    return json.loads((Path(__file__).parents[2]/'config/model.json').read_text())


def test_refined_reconstruction_preserves_measured_sites_and_straight_lengths():
    p=payload()['cable'];p['segments_per_marker_interval']=4
    cable=CableConfiguration.from_mapping(p)
    arc=np.r_[0,np.cumsum(cable.marker_interval_lengths_m)]
    sites=np.zeros((3,11,3));sites[:,:,2]=-arc
    sites[:,:,0]=np.arange(3)[:,None]*.1
    for kind in ['linear','cubic']:
        q=reconstruct(sites,cable,kind)
        np.testing.assert_allclose(q[:,cable.marker_node_indices],sites,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(q,axis=1),axis=-1),
                                   np.broadcast_to(cable.rest_lengths_m,(3,40)),atol=1e-12)


def test_boundary_assembly_has_correct_attachment_and_does_not_move_markers_in_raw_truth():
    p=payload();cable=CableConfiguration.from_mapping(p['cable'])
    markers=np.zeros((241,10,3));markers[:,:,2]=-np.cumsum(cable.marker_interval_lengths_m)
    position=np.zeros((241,3));position[:,2]=.055
    processed={'uav_position_m':position,'cable_marker_positions_m':markers}
    data={'synthetic':dict(processed=processed,rotation=np.broadcast_to(np.eye(3),(241,3,3)),
                           role='training',windows=[(20,{'intensity':'low'})])}
    params={'EI_n_m2':2e-6,'Cb_n_m2_s':1e-4}
    for variant in [{'name':'base'},{'name':'clamp','boundary':'body_clamp'}]:
        _,_,mask,q,v,b,truth,_=assemble(data,variant,p,params)
        torch.testing.assert_close(q[:,:mask[0]],b[:,0])
        np.testing.assert_array_equal(truth[0],markers[20:221])
        assert torch.isfinite(q).all() and torch.isfinite(v).all()
