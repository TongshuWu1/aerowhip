"""Fixed-shape CUDA evaluator for the unchanged MPPI strike objective."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import threading

import torch


FLOAT_FIELDS = (
    "constraint_violation",
    "impact_time_s",
    "position_error_m",
    "directional_speed_m_s",
    "tip_speed_m_s",
    "direction_cosine",
    "direction_error_deg",
    "drone_displacement_at_impact_m",
    "maximum_drone_excursion_m",
    "minimum_drone_clearance_m",
    "maximum_drone_speed_m_s",
    "minimum_scene_height_m",
    "maximum_drone_altitude_m",
    "minimum_cable_drone_clearance_m",
    "minimum_non_tip_target_distance_m",
    "maximum_acceleration_m_s2",
    "acceleration_effort",
    "acceleration_smoothness",
    "position_cost",
    "speed_cost",
    "predictive_speed_cost",
    "direction_cost",
    "success_cost",
    "drone_displacement_cost",
    "safety_cost",
    "control_cost",
    "workspace_violation",
    "keepout_violation",
    "speed_limit_violation",
    "ground_violation",
    "altitude_violation",
    "cable_drone_violation",
    "non_tip_contact_violation",
    "actuator_violation",
    "mppi_objective",
)


_SOURCE = r"""
extern "C" __device__ __forceinline__ float norm3_values(
    float x, float y, float z
) {
    return sqrtf(x * x + y * y + z * z);
}

extern "C" __global__ void fixed_mppi_cost_11node(
    const float* __restrict__ time_s,
    const float* __restrict__ drone_positions,
    const float* __restrict__ drone_velocities,
    const float* __restrict__ cable_positions,
    const float* __restrict__ cable_velocities,
    const float* __restrict__ accelerations,
    const float* __restrict__ initial_drone,
    const float* __restrict__ target,
    const float* __restrict__ direction,
    float* __restrict__ output,
    long long* __restrict__ impact_output,
    unsigned char* __restrict__ boolean_output,
    int batch,
    int frames,
    int nodes,
    int controls,
    int initial_batch,
    int steps_per_control,
    int objective_stage,
    int enforce_workspace,
    float target_radius,
    float minimum_impact_speed,
    float cone_cosine,
    float position_weight,
    float position_sigma,
    float velocity_gate_sigma,
    float speed_weight,
    float predictive_gate_sigma,
    float predictive_speed_ratio,
    float predictive_speed_weight,
    float direction_weight,
    float success_magnitude,
    float displacement_weight,
    float safety_weight,
    float maximum_drone_excursion_limit,
    float drone_keepout_radius,
    float maximum_drone_speed_limit,
    float ground_height,
    float ground_clearance,
    float maximum_altitude,
    float cable_drone_clearance_limit,
    float maximum_acceleration_limit,
    float effort_weight,
    float smoothness_weight
) {
    const int sample = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (sample >= batch) return;
    constexpr float INF = 3.402823466e+38f;
    constexpr float RAD_TO_DEG = 57.29577951308232f;
    const long long drone_base = (long long)sample * frames * 3;
    const long long cable_base = (long long)sample * frames * nodes * 3;
    const long long acceleration_base = (long long)sample * controls * 3;

    bool has_contact = false;
    int first_contact_frame = 1;
    int closest_frame = 1;
    float closest_distance = INF;
    for (int frame = 1; frame < frames; ++frame) {
        const long long tip = cable_base + ((long long)frame * nodes + nodes - 1) * 3;
        const float dx = cable_positions[tip] - target[0];
        const float dy = cable_positions[tip + 1] - target[1];
        const float dz = cable_positions[tip + 2] - target[2];
        const float distance = norm3_values(dx, dy, dz);
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_frame = frame;
        }
        if (!has_contact && distance <= target_radius) {
            has_contact = true;
            first_contact_frame = frame;
        }
    }
    const int impact_frame = has_contact ? first_contact_frame : closest_frame;
    const long long tip = cable_base + ((long long)impact_frame * nodes + nodes - 1) * 3;
    const float tip_dx = cable_positions[tip] - target[0];
    const float tip_dy = cable_positions[tip + 1] - target[1];
    const float tip_dz = cable_positions[tip + 2] - target[2];
    const float distance = norm3_values(tip_dx, tip_dy, tip_dz);
    const float vx = cable_velocities[tip];
    const float vy = cable_velocities[tip + 1];
    const float vz = cable_velocities[tip + 2];
    const float total_tip_speed = norm3_values(vx, vy, vz);
    const float directed_speed = vx * direction[0] + vy * direction[1] + vz * direction[2];
    float direction_cosine = directed_speed / fmaxf(total_tip_speed, 1.0e-9f);
    direction_cosine = fminf(1.0f, fmaxf(-1.0f, direction_cosine));
    const float direction_error_deg = acosf(direction_cosine) * RAD_TO_DEG;

    const int initial_index = initial_batch == 1 ? 0 : sample;
    const float initial_x = initial_drone[initial_index * 3];
    const float initial_y = initial_drone[initial_index * 3 + 1];
    const float initial_z = initial_drone[initial_index * 3 + 2];
    float maximum_drone_excursion = 0.0f;
    float minimum_drone_clearance = INF;
    float maximum_drone_speed = 0.0f;
    float minimum_scene_height = INF;
    float maximum_drone_altitude = -INF;
    float minimum_cable_drone_clearance = INF;
    for (int frame = 0; frame <= impact_frame; ++frame) {
        const long long drone = drone_base + (long long)frame * 3;
        const float px = drone_positions[drone];
        const float py = drone_positions[drone + 1];
        const float pz = drone_positions[drone + 2];
        maximum_drone_excursion = fmaxf(
            maximum_drone_excursion,
            norm3_values(px - initial_x, py - initial_y, pz - initial_z)
        );
        minimum_drone_clearance = fminf(
            minimum_drone_clearance,
            norm3_values(px - target[0], py - target[1], pz - target[2])
        );
        maximum_drone_speed = fmaxf(
            maximum_drone_speed,
            norm3_values(
                drone_velocities[drone],
                drone_velocities[drone + 1],
                drone_velocities[drone + 2]
            )
        );
        minimum_scene_height = fminf(minimum_scene_height, pz);
        maximum_drone_altitude = fmaxf(maximum_drone_altitude, pz);
        for (int node = 0; node < nodes; ++node) {
            const long long q = cable_base + ((long long)frame * nodes + node) * 3;
            minimum_scene_height = fminf(minimum_scene_height, cable_positions[q + 2]);
            if (node > 0) {
                minimum_cable_drone_clearance = fminf(
                    minimum_cable_drone_clearance,
                    norm3_values(
                        cable_positions[q] - px,
                        cable_positions[q + 1] - py,
                        cable_positions[q + 2] - pz
                    )
                );
            }
        }
    }
    const long long impact_drone = drone_base + (long long)impact_frame * 3;
    const float drone_displacement = norm3_values(
        drone_positions[impact_drone] - initial_x,
        drone_positions[impact_drone + 1] - initial_y,
        drone_positions[impact_drone + 2] - initial_z
    );

    float minimum_non_tip_distance = INF;
    for (int frame = 1; frame < impact_frame; ++frame) {
        for (int node = 0; node < nodes - 1; ++node) {
            const long long q = cable_base + ((long long)frame * nodes + node) * 3;
            minimum_non_tip_distance = fminf(
                minimum_non_tip_distance,
                norm3_values(
                    cable_positions[q] - target[0],
                    cable_positions[q + 1] - target[1],
                    cable_positions[q + 2] - target[2]
                )
            );
        }
    }

    int event_control = (impact_frame - 1) / steps_per_control;
    event_control = event_control < controls ? event_control : controls - 1;
    float maximum_acceleration = 0.0f;
    float effort = 0.0f;
    float smoothness = 0.0f;
    for (int control = 0; control <= event_control; ++control) {
        const long long u = acceleration_base + (long long)control * 3;
        const float ax = accelerations[u];
        const float ay = accelerations[u + 1];
        const float az = accelerations[u + 2];
        maximum_acceleration = fmaxf(maximum_acceleration, norm3_values(ax, ay, az));
        effort += ax * ax + ay * ay + az * az;
        if (control > 0) {
            const float dax = ax - accelerations[u - 3];
            const float day = ay - accelerations[u - 2];
            const float daz = az - accelerations[u - 1];
            smoothness += dax * dax + day * day + daz * daz;
        }
    }

    const float distance_squared = distance * distance;
    const float position_cost = position_weight * distance_squared
        / (distance_squared + position_sigma * position_sigma);
    const float proximity = expf(
        -distance_squared / (2.0f * velocity_gate_sigma * velocity_gate_sigma)
    );
    const float speed_deficit = fmaxf(0.0f, minimum_impact_speed - directed_speed);
    float speed_cost = speed_weight * proximity * speed_deficit * speed_deficit;
    const float predictive_proximity = expf(
        -distance_squared / (2.0f * predictive_gate_sigma * predictive_gate_sigma)
    );
    const float predictive_deficit = fmaxf(
        0.0f, predictive_speed_ratio * minimum_impact_speed - directed_speed
    );
    float predictive_speed_cost = predictive_speed_weight * predictive_proximity
        * predictive_deficit * predictive_deficit;
    const float direction_deficit = fmaxf(0.0f, cone_cosine - direction_cosine);
    float direction_cost = direction_weight * proximity
        * direction_deficit * direction_deficit;
    if (objective_stage == 0) {
        speed_cost = 0.0f;
        predictive_speed_cost = 0.0f;
        direction_cost = 0.0f;
    } else if (objective_stage == 1) {
        direction_cost = 0.0f;
    }
    const float displacement_cost = displacement_weight
        * drone_displacement * drone_displacement;
    const float workspace_violation = enforce_workspace
        ? powf(fmaxf(0.0f, maximum_drone_excursion - maximum_drone_excursion_limit)
            / maximum_drone_excursion_limit, 2.0f)
        : 0.0f;
    const float keepout_violation = powf(
        fmaxf(0.0f, drone_keepout_radius - minimum_drone_clearance)
            / drone_keepout_radius, 2.0f
    );
    const float speed_limit_violation = powf(
        fmaxf(0.0f, maximum_drone_speed - maximum_drone_speed_limit)
            / maximum_drone_speed_limit, 2.0f
    );
    const float ground_violation = powf(
        fmaxf(0.0f, ground_height + ground_clearance - minimum_scene_height)
            / ground_clearance, 2.0f
    );
    const float altitude_violation = powf(
        fmaxf(0.0f, maximum_drone_altitude - maximum_altitude)
            / maximum_altitude, 2.0f
    );
    const float cable_drone_violation = powf(
        fmaxf(0.0f, cable_drone_clearance_limit - minimum_cable_drone_clearance)
            / cable_drone_clearance_limit, 2.0f
    );
    const float non_tip_violation = powf(
        fmaxf(0.0f, target_radius - minimum_non_tip_distance) / target_radius,
        2.0f
    );
    const float actuator_violation = powf(
        fmaxf(0.0f, maximum_acceleration - maximum_acceleration_limit)
            / maximum_acceleration_limit, 2.0f
    );
    const float safety_violation = workspace_violation + keepout_violation
        + speed_limit_violation + ground_violation + altitude_violation
        + cable_drone_violation + non_tip_violation + actuator_violation;
    const float safety_cost = safety_weight * safety_violation;
    const bool position_success = has_contact;
    const bool speed_success = directed_speed >= minimum_impact_speed;
    const bool direction_success = direction_cosine >= cone_cosine;
    const bool task_success = objective_stage == 0
        ? position_success
        : (objective_stage == 1
            ? position_success && speed_success
            : position_success && speed_success && direction_success);
    const bool safe = safety_violation == 0.0f;
    const float success_cost = task_success && safe ? -success_magnitude : 0.0f;
    const float control_cost = effort_weight * effort + smoothness_weight * smoothness;
    const float total_cost = position_cost + speed_cost + predictive_speed_cost
        + direction_cost + success_cost + displacement_cost + safety_cost + control_cost;
    const float position_violation = powf(
        fmaxf(0.0f, distance - target_radius) / target_radius, 2.0f
    );
    const float speed_violation = powf(
        fmaxf(0.0f, minimum_impact_speed - directed_speed) / minimum_impact_speed,
        2.0f
    );
    float task_violation = position_violation;
    if (objective_stage >= 1) task_violation += speed_violation;
    if (objective_stage == 2) task_violation += direction_deficit * direction_deficit;
    const float constraint_violation = task_violation + safety_violation;

    float* out = output + (long long)sample * 35;
    out[0] = constraint_violation;
    out[1] = time_s[impact_frame];
    out[2] = distance;
    out[3] = directed_speed;
    out[4] = total_tip_speed;
    out[5] = direction_cosine;
    out[6] = direction_error_deg;
    out[7] = drone_displacement;
    out[8] = maximum_drone_excursion;
    out[9] = minimum_drone_clearance;
    out[10] = maximum_drone_speed;
    out[11] = minimum_scene_height;
    out[12] = maximum_drone_altitude;
    out[13] = minimum_cable_drone_clearance;
    out[14] = minimum_non_tip_distance;
    out[15] = maximum_acceleration;
    out[16] = effort;
    out[17] = smoothness;
    out[18] = position_cost;
    out[19] = speed_cost;
    out[20] = predictive_speed_cost;
    out[21] = direction_cost;
    out[22] = success_cost;
    out[23] = displacement_cost;
    out[24] = safety_cost;
    out[25] = control_cost;
    out[26] = workspace_violation;
    out[27] = keepout_violation;
    out[28] = speed_limit_violation;
    out[29] = ground_violation;
    out[30] = altitude_violation;
    out[31] = cable_drone_violation;
    out[32] = non_tip_violation;
    out[33] = actuator_violation;
    out[34] = total_cost;
    impact_output[sample] = (long long)impact_frame;
    boolean_output[(long long)sample * 3] = (unsigned char)(task_success && safe);
    boolean_output[(long long)sample * 3 + 1] = (unsigned char)has_contact;
    boolean_output[(long long)sample * 3 + 2] = 0;
}
"""


class _Kernel:
    def __init__(self) -> None:
        cuda_path = Path(os.environ.get("CUDA_PATH", ""))
        bin_path = cuda_path / "bin"
        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(bin_path))
        candidates = sorted(bin_path.glob("nvrtc64_*.dll"))
        if not candidates:
            raise RuntimeError("NVRTC is required for the fixed MPPI evaluator.")
        self.nvrtc = ctypes.WinDLL(str(candidates[-1]))
        self.cuda = ctypes.WinDLL("nvcuda.dll")
        self.nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        self.cuda.cuGetErrorString.restype = ctypes.c_int
        torch.cuda.current_stream()
        self._check_cuda(self.cuda.cuInit(0), "cuInit")
        program = ctypes.c_void_p()
        self._check_nvrtc(
            self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(program), _SOURCE.encode(), b"fixed_mppi_cost.cu", 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        try:
            major, minor = torch.cuda.get_device_capability()
            options_data = [
                f"--gpu-architecture=compute_{major}{minor}".encode(),
                b"--std=c++17",
                b"--fmad=true",
            ]
            options = (ctypes.c_char_p * len(options_data))(*options_data)
            code = self.nvrtc.nvrtcCompileProgram(program, len(options_data), options)
            if code:
                size = ctypes.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(log.value.decode("utf-8", "replace"))
            size = ctypes.c_size_t()
            self._check_nvrtc(self.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
            ptx = ctypes.create_string_buffer(size.value)
            self._check_nvrtc(self.nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        self.module = ctypes.c_void_p()
        self._check_cuda(self.cuda.cuModuleLoadDataEx(ctypes.byref(self.module), ptx, 0, None, None), "cuModuleLoadDataEx")
        self.function = ctypes.c_void_p()
        self._check_cuda(
            self.cuda.cuModuleGetFunction(ctypes.byref(self.function), self.module, b"fixed_mppi_cost_11node"),
            "cuModuleGetFunction",
        )

    def _check_nvrtc(self, code: int, operation: str) -> None:
        if code:
            raise RuntimeError(
                f"{operation}: {self.nvrtc.nvrtcGetErrorString(code).decode()}"
            )

    def _check_cuda(self, code: int, operation: str) -> None:
        if code:
            message = ctypes.c_char_p()
            self.cuda.cuGetErrorString(code, ctypes.byref(message))
            raise RuntimeError(f"{operation}: {message.value.decode() if message.value else code}")

    def launch(self, rollout, initial_state, problem, simulator, settings):
        batch = rollout.batch_size
        output = torch.empty((batch, len(FLOAT_FIELDS)), dtype=torch.float32, device=rollout.cable_positions_m.device)
        impact = torch.empty((batch,), dtype=torch.int64, device=output.device)
        booleans = torch.empty((batch, 3), dtype=torch.bool, device=output.device)
        target = torch.as_tensor(problem.target_position_m, dtype=torch.float32, device=output.device)
        direction = torch.as_tensor(problem.impact_direction, dtype=torch.float32, device=output.device)
        tensors = (
            rollout.time_s,
            rollout.drone_positions_m,
            rollout.drone_velocities_m_s,
            rollout.cable_positions_m,
            rollout.cable_velocities_m_s,
            rollout.accelerations_m_s2,
            initial_state.drone_position_m,
            target,
            direction,
            output,
            impact,
            booleans,
        )
        # Retain any contiguous copies until the asynchronous kernel launch has
        # consumed them.  Most production inputs are already contiguous, but a
        # temporary returned by ``contiguous()`` must not be allowed to die
        # while the current CUDA stream is still using its storage.
        contiguous_tensors = tuple(value.contiguous() for value in tensors)
        argument_values: list[object] = [
            ctypes.c_void_p(value.data_ptr()) for value in contiguous_tensors
        ]
        stage = {"position": 0, "speed": 1, "full": 2}[settings.objective_stage]
        argument_values += [
            ctypes.c_int(batch), ctypes.c_int(rollout.frame_count),
            ctypes.c_int(rollout.cable_positions_m.shape[2]),
            ctypes.c_int(rollout.accelerations_m_s2.shape[1]),
            ctypes.c_int(initial_state.drone_position_m.shape[0]),
            ctypes.c_int(simulator.settings.steps_per_control), ctypes.c_int(stage),
            ctypes.c_int(int(settings.enforce_workspace_limit)),
        ]
        float_values = (
            problem.maximum_tip_error_m, problem.minimum_impact_speed_m_s,
            math.cos(math.radians(problem.maximum_impact_angle_deg)),
            settings.position_weight, settings.position_sigma_m,
            settings.velocity_gate_sigma_m, settings.speed_weight,
            settings.predictive_velocity_gate_sigma_m, settings.predictive_speed_ratio,
            settings.predictive_speed_weight, settings.direction_weight,
            settings.success_cost, settings.drone_displacement_weight,
            settings.safety_weight, problem.maximum_drone_excursion_m,
            problem.drone_keepout_radius_m, simulator.settings.maximum_speed_m_s,
            settings.ground_height_m, settings.ground_clearance_m,
            settings.maximum_altitude_m, settings.cable_drone_clearance_m,
            simulator.settings.maximum_acceleration_m_s2,
            settings.control_effort_weight, settings.control_smoothness_weight,
        )
        argument_values += [ctypes.c_float(value) for value in float_values]
        arguments = (ctypes.c_void_p * len(argument_values))(*[
            ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in argument_values
        ])
        stream = ctypes.c_void_p(torch.cuda.current_stream(output.device).cuda_stream)
        block = 128
        self._check_cuda(
            self.cuda.cuLaunchKernel(
                self.function, (batch + block - 1) // block, 1, 1,
                block, 1, 1, 0, stream, arguments, None
            ),
            "cuLaunchKernel(fixed_mppi_cost_11node)",
        )
        diagnostics = {name: output[:, index] for index, name in enumerate(FLOAT_FIELDS)}
        diagnostics.update(
            {
                "impact_frame": impact,
                "feasible": booleans[:, 0],
                "geometric_tip_contact": booleans[:, 1],
                "physical_tip_contact": booleans[:, 2],
            }
        )
        return diagnostics["mppi_objective"], diagnostics


_lock = threading.Lock()
_kernel: _Kernel | None = None


def evaluate_fixed_mppi_cost(rollout, initial_state, problem, simulator, settings):
    global _kernel
    if _kernel is None:
        with _lock:
            if _kernel is None:
                _kernel = _Kernel()
    return _kernel.launch(rollout, initial_state, problem, simulator, settings)
