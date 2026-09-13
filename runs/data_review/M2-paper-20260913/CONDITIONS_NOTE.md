# Initial hover and cable-state mismatch

The operator clarified that battery condition and damage are not considered the explanation, and reports small cable-tip motion during the initial hover, attributed to propeller airflow. The command-correction code initializes every rollout with the same frozen M0 cable shape and zero nodal velocity, rather than the measured pre-strike state of each take. Variation in the actual initial cable state is therefore a plausible source of tracking variability. Pre-strike motion and its association with strike error have not yet been quantified; neither airflow causation nor its contribution to the M1-to-M2 difference is established.

The earlier battery/damage explanation has been superseded by this operator clarification; its history is retained in reported_conditions.json. All five takes and all measured results remain unchanged. This is a limitation and a hypothesis, not a demonstrated cause of M2 performance.
