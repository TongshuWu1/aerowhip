"""Field tooltips for reward settings and hit conditions."""

FIELD_HELP = {
    "maximum_displacement_weight": "Penalizes increases in the drone's maximum distance from its starting position, on hits and misses. Returning does not erase the cost, and repeating the same excursion does not add it again.",
    "progress_weight": "Total budget for improving the closest tip distance. Repeating an unchanged approach earns no extra points.",
    "strike_quality_improvement_weight": "Total budget for improving speed near the target, projected along the desired strike direction. The advanced speed-reference setting selects world velocity or velocity relative to the drone.",
    "point_displacement_integral_weight": "Charged throughout the attempt: weight × integral of the scaled displacement cost. Larger and longer attachment movements cost more.",
    "time_to_success_weight_per_s": "Points charged per second until a valid hit, timeout, or numerical failure.",
    "success_bonus": "Paid once, only when the tip passes all hit conditions.",
    "success_forward_return_bonus_weight": "Maximum bonus on a valid hit for bringing the attachment back from its furthest forward position. The actual amount depends on the trajectory.",
    "success_release_bonus_weight": "Maximum bonus on a valid hit when the attachment moves backward and the tip moves forward relative to it. The actual amount depends on the trajectory.",
    "terminal_displacement_weight": "Only charged on a valid hit: weight × log(1 + (displacement / scale)²). This is a multiplier, not a fixed points deduction.",
    "non_tip_first_penalty": "Charged once when another cable marker reaches the target first. That contact also prevents a later success in the same attempt.",
    "invalid_tip_entry_penalty": "Charged once for an invalid tip contact. With the first-contact rule enabled, the simulated attempt ends even when this penalty is zero. Real force execution still ends at its preplanned cutoff.",
    "timeout_penalty": "Extra cost at the time limit, in addition to the elapsed-time and displacement costs.",
    "numerical_failure_penalty": "Emergency terminal cost for a non-finite state or a state exceeding numerical limits.",
    "proximity_scale_m": "How quickly speed credit falls with distance from the target. This is not the hit radius.",
    "directed_speed_reward_cap_m_s": "Forward tip speed at which shaping saturates, using the selected speed reward reference. The hit threshold always uses world speed.",
    "forward_excursion_scale_m": "How much forward attachment motion activates return and release bonuses.",
    "point_backward_speed_scale_m_s": "Sets the attachment-backward-speed response of the release bonus.",
    "relative_tip_forward_speed_scale_m_s": "Sets the relative-forward-tip-speed response of the release bonus.",
    "displacement_cost_scale_m": "Smaller scales make the same displacement more expensive, during motion and at a hit.",
}

HIT_FIELDS = (
    ("tip_target_distance_m", "Target radius", " m", 0.001, 1.0, "The tip's path must enter this sphere around the target."),
    ("minimum_directed_tip_speed_m_s", "Min. forward speed", " m/s", 0.0, 30.0, "World-frame tip velocity projected onto the desired strike direction. This is not total speed or speed relative to the attachment."),
    ("maximum_tip_velocity_to_desired_direction_error_deg", "Max. velocity angle", "°", 0.0, 180.0, "Angle between world-frame tip velocity and the desired strike direction. Smaller means a straighter strike. The cable tangent is not used."),
)
