import numpy as np
import pytest
from experimental_data.whip_validation import verify_split,command_inputs,errors


def test_either_model_using_held_out_take_is_rejected():
    verify_split('c',['a','b'],['a','b'])
    for cable,drone in [(['c'],['a']),(['a'],['c'])]:
        with pytest.raises(ValueError,match='entered model fitting'):verify_split('c',cable,drone)


def test_commands_use_logged_hold_and_initial_offset_only():
    a={'commands':np.zeros((100,9)),'quaternion':np.ones((100,4))}
    a['commands'][30:40,6]=2.
    first=command_inputs(a,30,30,6,np.array([.01,0.,-.05]))
    a['quaternion'][31:]=999 # future measured attitude is not an input
    second=command_inputs(a,30,30,6,np.array([.01,0.,-.05]))
    for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
    assert first[0][6,6]==2 and first[0][16,6]==0


def test_invalid_measurements_are_masked_not_filled():
    actual=np.zeros((3,3));pred=actual.copy();pred[1]=100
    assert errors(pred,actual,np.array([True,False,True]))['rmse_m']==0
