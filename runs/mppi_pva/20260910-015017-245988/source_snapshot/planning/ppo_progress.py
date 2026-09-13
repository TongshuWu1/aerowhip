"""Shared checkpoint/plateau comparison for training and its dashboard."""
import math


def evaluation_better(score,success,old_score,old_success,*,relative=None,success_priority=False):
    if not math.isfinite(score) or not math.isfinite(success):return False
    if success_priority and success!=old_success:return success>old_success
    if not math.isfinite(old_score):return True
    threshold=0 if relative is None else max(.1,abs(old_score)*relative)
    return score>old_score+threshold
