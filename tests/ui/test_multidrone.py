"""Coordinate/provenance and UI checks for the separate Isaac presentation."""
import hashlib
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from tools.multidrone_data import grid_offsets, display_frame, load_replay


def fixture_data():
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_rotvec([[.4,.2,.1],[-.2,.3,0]]).as_matrix()[None]
    rotation=np.repeat(rotation,3,axis=0)
    origin=np.array([[[1,2,3],[-1,2,3]]]*3,dtype=float)
    offset=np.array([.006,-.012,-.055])
    root=origin+np.einsum('tbij,j->tbi',rotation,offset)
    return dict(origin=origin,rotation=rotation,cable=np.stack([root,root-[0,0,.9]],axis=2),
        target=np.array([[1,0,1.4],[1,.04,1.4]]),time_s=np.arange(3)/150,attachment_offset_m=offset)


def test_display_grid_preserves_relative_geometry_and_pose():
    data=fixture_data();original={k:v.copy() for k,v in data.items()}
    p,r,q,t=display_frame(data,1,2)
    np.testing.assert_allclose(q[:,:,0:3]-p[:,None],data['cable'][1]-data['origin'][1,:,None],atol=1e-14)
    np.testing.assert_allclose(q[:,0],p+np.einsum('bij,j->bi',r,data['attachment_offset_m']),atol=1e-14)
    np.testing.assert_allclose(t-p,data['target']-data['origin'][1],atol=1e-14)
    for k in data:np.testing.assert_array_equal(data[k],original[k])
    grid=grid_offsets(1024)
    assert len(np.unique(grid,axis=0))==1024
    assert np.max(grid[:,0])-np.min(grid[:,0])==124


def test_replay_rejects_corruption_and_detached_cable(tmp_path):
    data=fixture_data()
    def save():
        np.savez(tmp_path/'replay.npz',**data)
        (tmp_path/'replay.json').write_text(json.dumps(dict(schema='multidrone_replay_v1',num_envs=2,
            replay_sha256=hashlib.sha256((tmp_path/'replay.npz').read_bytes()).hexdigest())))
    save();load_replay(tmp_path)
    with (tmp_path/'replay.npz').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='provenance'):load_replay(tmp_path)
    data['origin'][0,0,0]+=.1;save()
    with pytest.raises(ValueError,match='attachment disagree'):load_replay(tmp_path)


def test_video_encodes_all_fixed_time_frames(tmp_path):
    from PIL import Image
    import imageio_ffmpeg
    from tools.encode_multidrone_video import encode
    for i in range(3):Image.new('RGB',(64,48),(i*70,20,30)).save(tmp_path/f'frame_{i:05d}.png')
    (tmp_path/'recording.json').write_text(json.dumps(dict(frame_count=3,fps=30,label='Deterministic checkpoint replay')))
    result=encode(tmp_path)
    reader=imageio_ffmpeg.read_frames(str(result),pix_fmt='rgb24')
    meta=next(reader)
    assert tuple(meta['size'])==(64,48) and meta['fps']==30
    assert len(list(reader))==3
    with pytest.raises(ValueError,match='already exists'):encode(tmp_path)
