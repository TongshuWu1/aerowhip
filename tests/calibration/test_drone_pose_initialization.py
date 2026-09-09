import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experimental_data.drone_pose_initialization import initialize_drone_pose


def observations():
    t=np.linspace(0,1,101)
    p=np.column_stack([t*t,.3*t,1.5+.1*t*t])
    r0=Rotation.from_euler('xyz',[.2,-.15,.4])
    rotations=r0*Rotation.from_rotvec(np.column_stack([0*t,.6*t*t,0*t]))
    return t,p,rotations.as_quat()


def initialize(t,p,q,**kwargs):
    return initialize_drone_pose(t,p,q,cutoff_s=.705,offset_tracking_m=[.007,-.014,-.055],**kwargs)


def test_endpoint_linear_and_angular_derivatives_and_attachment():
    t,p,q=observations();s=initialize(t,p,q)
    assert s['time_s']==pytest.approx(.7)
    np.testing.assert_allclose(s['velocity_origin_m_s'],[1.4,.3,.14],atol=1e-12)
    np.testing.assert_allclose(s['omega_tracking_rad_s'],[0,.84,0],atol=1e-12)
    r=np.array([.007,-.014,-.055]);R=Rotation.from_quat(q[70]).as_matrix()
    np.testing.assert_allclose(s['position_attachment_m'],p[70]+R@r,atol=1e-12)
    # Independently differentiate the analytic rotating attachment trajectory.
    def attachment_at(t):
        rotation=Rotation.from_euler('xyz',[.2,-.15,.4])*Rotation.from_rotvec([0,.6*t*t,0])
        return np.array([t*t,.3*t,1.5+.1*t*t])+rotation.apply(r)
    h=1e-5
    np.testing.assert_allclose(s['velocity_attachment_m_s'],(attachment_at(.7+h)-attachment_at(.7-h))/(2*h),atol=1e-9)
    assert s['controller_memory_observed'] is False


def test_future_changes_and_quaternion_signs_cannot_change_initial_state():
    t,p,q=observations();expected=initialize(t,p,q)
    p[t>=.705]=np.nan;q[t>=.705]=np.nan
    q[::2]*=-3
    actual=initialize(t,p,q)
    for name in ['position_origin_m','velocity_origin_m_s','rotation_tracking_to_world',
                 'omega_tracking_rad_s','position_attachment_m','velocity_attachment_m_s']:
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-12)


@pytest.mark.parametrize('damage',['gap','invalid_pose','stale','insufficient','invalid_position'])
def test_missing_recent_history_is_not_silently_filled(damage):
    t,p,q=observations()
    if damage=='gap':
        t=np.delete(t,[65,66,67]);p=np.delete(p,[65,66,67],axis=0);q=np.delete(q,[65,66,67],axis=0)
    if damage=='invalid_pose':q[65]=0
    if damage=='invalid_position':p[65]=np.nan
    if damage=='stale':t[t<.705]-=.1
    if damage=='insufficient':t=t[:5];p=p[:5];q=q[:5]
    with pytest.raises(ValueError):initialize(t,p,q)
