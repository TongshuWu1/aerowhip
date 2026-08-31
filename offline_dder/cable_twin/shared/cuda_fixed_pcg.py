"""NVRTC fixed-topology CUDA operators for online 11/21/31-node DDER.

The implementation intentionally owns no physics.  It receives the same
state and physical arrays as the reference PyTorch path and performs the same
curvature Jacobian, float64 zero-initialized Jacobi-PCG recurrence, and
four-position-plus-one-velocity projection. Unsupported shapes fall back at
the call site.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
from typing import Literal

import torch


_SOURCE = r"""
extern "C" __device__ __forceinline__ double warp_sum(double value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ void fixed_pcg_30x60(
    const double* __restrict__ systems,
    const double* __restrict__ right_hand_sides,
    double* __restrict__ solutions,
    int batch_size
) {
    constexpr int D = 30;
    constexpr int ITERATIONS = 60;
    const int system_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (system_index >= batch_size) return;

    __shared__ double matrix[D * D];
    __shared__ double direction_shared[32];
    const long long matrix_base = (long long)system_index * D * D;
    for (int index = lane; index < D * D; index += 32) {
        matrix[index] = systems[matrix_base + index];
    }
    __syncwarp();

    const long long vector_base = (long long)system_index * D;
    double solution = 0.0;
    double residual = lane < D ? right_hand_sides[vector_base + lane] : 0.0;
    const double diagonal = lane < D ? matrix[lane * D + lane] : 1.0;
    double preconditioned = lane < D ? residual / diagonal : 0.0;
    double direction = preconditioned;
    double residual_product = warp_sum(residual * preconditioned);
    constexpr double TINY = 2.2250738585072014e-308;

#pragma unroll 1
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        direction_shared[lane] = direction;
        __syncwarp();
        double applied = 0.0;
        if (lane < D) {
#pragma unroll
            for (int column = 0; column < D; ++column) {
                applied += matrix[lane * D + column] * direction_shared[column];
            }
        }
        const double denominator = warp_sum(direction * applied);
        const double alpha = fabs(denominator) > TINY
            ? residual_product / denominator
            : 0.0;
        solution += alpha * direction;
        residual -= alpha * applied;
        preconditioned = lane < D ? residual / diagonal : 0.0;
        const double next_product = warp_sum(residual * preconditioned);
        const double beta = fabs(residual_product) > TINY
            ? next_product / residual_product
            : 0.0;
        direction = preconditioned + beta * direction;
        residual_product = next_product;
    }

    if (lane < D) solutions[vector_base + lane] = solution;
}

extern "C" __device__ __forceinline__ void skew3(
    const float v[3], float result[9]
) {
    result[0] = 0.0f; result[1] = -v[2]; result[2] = v[1];
    result[3] = v[2]; result[4] = 0.0f; result[5] = -v[0];
    result[6] = -v[1]; result[7] = v[0]; result[8] = 0.0f;
}

extern "C" __device__ __forceinline__ void multiply3(
    const float left[9], const float right[9], float result[9]
) {
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
            float value = 0.0f;
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                value += left[row * 3 + inner] * right[inner * 3 + column];
            }
            result[row * 3 + column] = value;
        }
    }
}

extern "C" __device__ __forceinline__ float dot3(
    const float a[3], const float b[3]
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

extern "C" __device__ __forceinline__ void cross3(
    const float a[3], const float b[3], float result[3]
) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

extern "C" __device__ __forceinline__ void tangent_block(
    const float edge[3], float tangent[3], float result[9]
) {
    constexpr float EPSILON = 7.62939453125e-6f;
    const float length = fmaxf(sqrtf(dot3(edge, edge)), EPSILON);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) tangent[axis] = edge[axis] / length;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                ((row == column ? 1.0f : 0.0f) - tangent[row] * tangent[column])
                / length;
        }
    }
}

extern "C" __device__ __forceinline__ void objective_derivatives(
    const float previous[3],
    const float following[3],
    float derivative_previous[9],
    float derivative_following[9]
) {
    constexpr float EPSILON = 7.62939453125e-6f;
    constexpr float DENOMINATOR_EPSILON = 1.52587890625e-5f;
    float cross[3];
    cross3(previous, following, cross);
    const float cosine = dot3(previous, following);
    const float denominator = fmaxf(1.0f + cosine, DENOMINATOR_EPSILON);
    const float denominator_squared = denominator * denominator;
    float previous_skew[9], following_skew[9];
    skew3(previous, previous_skew);
    skew3(following, following_skew);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
            const int index = row * 3 + column;
            derivative_previous[index] = 2.0f * (
                -following_skew[index] / denominator
                - cross[row] * following[column] / denominator_squared
            );
            derivative_following[index] = 2.0f * (
                previous_skew[index] / denominator
                - cross[row] * previous[column] / denominator_squared
            );
        }
    }

    float tangent_sum[3], tangent_difference[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        tangent_sum[axis] = previous[axis] + following[axis];
        tangent_difference[axis] = previous[axis] - following[axis];
    }
    const float sum_norm = fmaxf(dot3(tangent_sum, tangent_sum), EPSILON);
    const float difference_norm = fmaxf(
        dot3(tangent_difference, tangent_difference), EPSILON
    );
    const float normal_norm = fmaxf(dot3(cross, cross), EPSILON);
    const float sum_eigenvalue = fmaxf(1.0f - cosine, EPSILON);
    const float difference_eigenvalue = fmaxf(1.0f + cosine, EPSILON);
    float angular_inverse[9];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
            const int index = row * 3 + column;
            const float general =
                tangent_sum[row] * tangent_sum[column]
                    / sum_norm / sum_eigenvalue
                + tangent_difference[row] * tangent_difference[column]
                    / difference_norm / difference_eigenvalue
                + 0.5f * cross[row] * cross[column] / normal_norm;
            const float straight = 0.5f * (
                (row == column ? 1.0f : 0.0f)
                - previous[row] * previous[column]
            );
            angular_inverse[index] = (1.0f - cosine) > 1.0e-7f
                ? general
                : straight;
        }
    }
    float spin_previous[9], spin_following[9];
    multiply3(angular_inverse, previous_skew, spin_previous);
    multiply3(angular_inverse, following_skew, spin_following);
    float curvature[3] = {
        2.0f * cross[0] / denominator,
        2.0f * cross[1] / denominator,
        2.0f * cross[2] / denominator,
    };
    float corotation[9], correction_previous[9], correction_following[9];
    skew3(curvature, corotation);
    multiply3(corotation, spin_previous, correction_previous);
    multiply3(corotation, spin_following, correction_following);
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        derivative_previous[index] += correction_previous[index];
        derivative_following[index] += correction_following[index];
    }
}

extern "C" __global__ void fixed_damping_11node_60pcg(
    const float* __restrict__ positions,
    const float* __restrict__ undamped_velocities,
    const float* __restrict__ boundary_velocities,
    const float* __restrict__ rest_lengths,
    const float* __restrict__ masses,
    const float* __restrict__ substep_dt,
    const float* __restrict__ damping,
    float* __restrict__ output_velocities,
    int batch_size
) {
    constexpr int N = 11;
    constexpr int D = 30;
    constexpr int R = 27;
    constexpr int BAND = 15;
    constexpr int ITERATIONS = 60;
    const int lane = (int)threadIdx.x;
    const int system_index = (int)blockIdx.x;
    if (system_index >= batch_size) return;

    __shared__ float jacobian[R * D];
    __shared__ float prescribed[R];
    // J has a three-node local stencil, so J^T W J couples only nodes at
    // graph distance <= 2.  BAND stores the exact five 3x3 block diagonals;
    // omitted dense entries are identically zero.
    __shared__ double matrix_band[D * BAND];
    __shared__ double right_hand_side[D];
    __shared__ double direction_shared[32];
    for (int index = lane; index < R * D; index += 32) jacobian[index] = 0.0f;
    for (int index = lane; index < R; index += 32) prescribed[index] = 0.0f;
    __syncwarp();

    const long long q_base = (long long)system_index * N * 3;
    if (lane == 0) {
#pragma unroll
        for (int joint = 0; joint < N - 2; ++joint) {
            float previous_edge[3], following_edge[3];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                previous_edge[axis] = positions[q_base + (joint + 1) * 3 + axis]
                    - positions[q_base + joint * 3 + axis];
                following_edge[axis] = positions[q_base + (joint + 2) * 3 + axis]
                    - positions[q_base + (joint + 1) * 3 + axis];
            }
            float previous[3], following[3];
            float previous_tangent_block[9], following_tangent_block[9];
            tangent_block(previous_edge, previous, previous_tangent_block);
            tangent_block(following_edge, following, following_tangent_block);
            float derivative_previous[9], derivative_following[9];
            objective_derivatives(
                previous, following, derivative_previous, derivative_following
            );
            float previous_block[9], following_block[9];
            multiply3(derivative_previous, previous_tangent_block, previous_block);
            multiply3(derivative_following, following_tangent_block, following_block);
#pragma unroll
            for (int row = 0; row < 3; ++row) {
                const int output_row = joint * 3 + row;
#pragma unroll
                for (int column = 0; column < 3; ++column) {
                    if (joint >= 1) {
                        jacobian[output_row * D + (joint - 1) * 3 + column]
                            -= previous_block[row * 3 + column];
                    }
                    jacobian[output_row * D + joint * 3 + column]
                        += previous_block[row * 3 + column]
                            - following_block[row * 3 + column];
                    jacobian[output_row * D + (joint + 1) * 3 + column]
                        += following_block[row * 3 + column];
                    if (joint == 0) {
                        prescribed[output_row] -= previous_block[row * 3 + column]
                            * boundary_velocities[(long long)system_index * 3 + column];
                    }
                }
            }
        }
    }
    __syncwarp();

    if (lane < D) {
        const int node = lane / 3 + 1;
        const int row_node = lane / 3;
        const float coefficient = substep_dt[system_index] * damping[system_index];
#pragma unroll
        for (int slot = 0; slot < BAND; ++slot) {
            const int node_delta = slot / 3 - 2;
            const int column_node = row_node + node_delta;
            const int column = column_node * 3 + slot % 3;
            float value = 0.0f;
            if (column_node >= 0 && column_node < 10) {
                float unit_damping = 0.0f;
#pragma unroll
                for (int row = 0; row < R; ++row) {
                    const int joint = row / 3;
                    const float weight = 2.0f
                        / (rest_lengths[joint] + rest_lengths[joint + 1]);
                    const float weighted = weight * jacobian[row * D + column];
                    unit_damping += jacobian[row * D + lane] * weighted;
                }
                value = coefficient * unit_damping;
                if (column == lane) value += masses[node];
            }
            matrix_band[lane * BAND + slot] = (double)value;
        }
        float prescribed_force = 0.0f;
#pragma unroll
        for (int row = 0; row < R; ++row) {
            const int joint = row / 3;
            const float weight = 2.0f
                / (rest_lengths[joint] + rest_lengths[joint + 1]);
            prescribed_force += jacobian[row * D + lane]
                * (weight * prescribed[row]);
        }
        prescribed_force *= -damping[system_index];
        const float rhs = masses[node]
            * undamped_velocities[q_base + (lane + 3)]
            + substep_dt[system_index] * prescribed_force;
        right_hand_side[lane] = (double)rhs;
    }
    __syncwarp();

    double solution = 0.0;
    double residual = lane < D ? right_hand_side[lane] : 0.0;
    const int diagonal_slot = 6 + lane % 3;
    const double diagonal = lane < D
        ? matrix_band[lane * BAND + diagonal_slot]
        : 1.0;
    double preconditioned = lane < D ? residual / diagonal : 0.0;
    double direction = preconditioned;
    double residual_product = warp_sum(residual * preconditioned);
    constexpr double TINY = 2.2250738585072014e-308;
#pragma unroll 1
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        direction_shared[lane] = direction;
        __syncwarp();
        double applied = 0.0;
        if (lane < D) {
            const int row_node = lane / 3;
#pragma unroll
            for (int slot = 0; slot < BAND; ++slot) {
                const int column_node = row_node + slot / 3 - 2;
                if (column_node >= 0 && column_node < 10) {
                    const int column = column_node * 3 + slot % 3;
                    applied += matrix_band[lane * BAND + slot]
                        * direction_shared[column];
                }
            }
        }
        const double denominator = warp_sum(direction * applied);
        const double alpha = fabs(denominator) > TINY
            ? residual_product / denominator : 0.0;
        solution += alpha * direction;
        residual -= alpha * applied;
        preconditioned = lane < D ? residual / diagonal : 0.0;
        const double next_product = warp_sum(residual * preconditioned);
        const double beta = fabs(residual_product) > TINY
            ? next_product / residual_product : 0.0;
        direction = preconditioned + beta * direction;
        residual_product = next_product;
    }

    if (lane < 3) {
        output_velocities[q_base + lane] = boundary_velocities[
            (long long)system_index * 3 + lane
        ];
    }
    if (lane < D) output_velocities[q_base + lane + 3] = (float)solution;
}

extern "C" __device__ __forceinline__ float safe_projection_denominator(
    float value
) {
    constexpr float EPSILON = 7.62939453125e-6f;
    if (fabsf(value) >= EPSILON) return value;
    return value < 0.0f ? -EPSILON : EPSILON;
}

extern "C" __global__ void fixed_projection_11node_4plus1(
    const float* __restrict__ predicted_positions,
    const float* __restrict__ previous_positions,
    const float* __restrict__ rest_lengths,
    const float* __restrict__ masses,
    const float* __restrict__ boundary_positions,
    const float* __restrict__ boundary_velocities,
    const float* __restrict__ substep_dt,
    float* __restrict__ output_positions,
    float* __restrict__ output_velocities,
    int batch_size
) {
    constexpr int N = 11;
    constexpr int E = 10;
    constexpr float EPSILON = 7.62939453125e-6f;
    constexpr float SOLVER_REGULARIZATION = 9.5367431640625e-7f;
    const int sample = (int)blockIdx.x;
    if (sample >= batch_size || threadIdx.x != 0) return;
    const long long base = (long long)sample * N * 3;
    const long long boundary_base = (long long)sample * 3;

    float value[N * 3];
    float velocity[N * 3];
    float inverse_mass[N];
    float direction[E * 3];
    float diagonal[E];
    float off_diagonal[E - 1];
    float right_hand_side[E];
    float modified_upper[E - 1];
    float modified_rhs[E];
    float multiplier[E];
#pragma unroll
    for (int index = 0; index < N * 3; ++index) {
        value[index] = predicted_positions[base + index];
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        value[axis] = boundary_positions[boundary_base + axis];
    }
    inverse_mass[0] = 0.0f;
#pragma unroll
    for (int node = 1; node < N; ++node) inverse_mass[node] = 1.0f / masses[node];

#pragma unroll
    for (int projection_iteration = 0; projection_iteration < 4; ++projection_iteration) {
#pragma unroll
        for (int edge = 0; edge < E; ++edge) {
            float delta[3];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                delta[axis] = value[(edge + 1) * 3 + axis] - value[edge * 3 + axis];
            }
            const float distance = sqrtf(dot3(delta, delta));
            const float denominator = fmaxf(distance, EPSILON);
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                direction[edge * 3 + axis] = delta[axis] / denominator;
            }
            diagonal[edge] = inverse_mass[edge] + inverse_mass[edge + 1];
            right_hand_side[edge] = -(distance - rest_lengths[edge]);
        }
        float maximum_diagonal = diagonal[0];
#pragma unroll
        for (int edge = 1; edge < E; ++edge) {
            maximum_diagonal = fmaxf(maximum_diagonal, diagonal[edge]);
        }
        const float regularization = SOLVER_REGULARIZATION * maximum_diagonal;
#pragma unroll
        for (int edge = 0; edge < E - 1; ++edge) {
            off_diagonal[edge] = -inverse_mass[edge + 1] * (
                direction[edge * 3] * direction[(edge + 1) * 3]
                + direction[edge * 3 + 1] * direction[(edge + 1) * 3 + 1]
                + direction[edge * 3 + 2] * direction[(edge + 1) * 3 + 2]
            );
        }
        float denominator = safe_projection_denominator(diagonal[0] + regularization);
        modified_rhs[0] = right_hand_side[0] / denominator;
        modified_upper[0] = off_diagonal[0] / denominator;
#pragma unroll
        for (int edge = 1; edge < E; ++edge) {
            denominator = safe_projection_denominator(
                diagonal[edge] + regularization
                - off_diagonal[edge - 1] * modified_upper[edge - 1]
            );
            modified_rhs[edge] = (
                right_hand_side[edge]
                - off_diagonal[edge - 1] * modified_rhs[edge - 1]
            ) / denominator;
            if (edge < E - 1) modified_upper[edge] = off_diagonal[edge] / denominator;
        }
        multiplier[E - 1] = modified_rhs[E - 1];
#pragma unroll
        for (int edge = E - 2; edge >= 0; --edge) {
            multiplier[edge] = modified_rhs[edge]
                - modified_upper[edge] * multiplier[edge + 1];
        }
#pragma unroll
        for (int edge = 0; edge < E; ++edge) {
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                const float correction = multiplier[edge] * direction[edge * 3 + axis];
                value[edge * 3 + axis] -= inverse_mass[edge] * correction;
                value[(edge + 1) * 3 + axis] += inverse_mass[edge + 1] * correction;
            }
        }
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            value[axis] = boundary_positions[boundary_base + axis];
        }
    }

    const float reciprocal_dt = 1.0f / substep_dt[sample];
#pragma unroll
    for (int index = 0; index < N * 3; ++index) {
        velocity[index] = (value[index] - previous_positions[base + index]) * reciprocal_dt;
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        velocity[axis] = boundary_velocities[boundary_base + axis];
    }
#pragma unroll
    for (int edge = 0; edge < E; ++edge) {
        float delta[3];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            delta[axis] = value[(edge + 1) * 3 + axis] - value[edge * 3 + axis];
        }
        const float distance = fmaxf(sqrtf(dot3(delta, delta)), EPSILON);
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            direction[edge * 3 + axis] = delta[axis] / distance;
        }
        diagonal[edge] = inverse_mass[edge] + inverse_mass[edge + 1];
        right_hand_side[edge] = -(
            direction[edge * 3]
                * (velocity[(edge + 1) * 3] - velocity[edge * 3])
            + direction[edge * 3 + 1]
                * (velocity[(edge + 1) * 3 + 1] - velocity[edge * 3 + 1])
            + direction[edge * 3 + 2]
                * (velocity[(edge + 1) * 3 + 2] - velocity[edge * 3 + 2])
        );
    }
    float maximum_diagonal = diagonal[0];
#pragma unroll
    for (int edge = 1; edge < E; ++edge) {
        maximum_diagonal = fmaxf(maximum_diagonal, diagonal[edge]);
    }
    const float regularization = SOLVER_REGULARIZATION * maximum_diagonal;
#pragma unroll
    for (int edge = 0; edge < E - 1; ++edge) {
        off_diagonal[edge] = -inverse_mass[edge + 1] * (
            direction[edge * 3] * direction[(edge + 1) * 3]
            + direction[edge * 3 + 1] * direction[(edge + 1) * 3 + 1]
            + direction[edge * 3 + 2] * direction[(edge + 1) * 3 + 2]
        );
    }
    float denominator = safe_projection_denominator(diagonal[0] + regularization);
    modified_rhs[0] = right_hand_side[0] / denominator;
    modified_upper[0] = off_diagonal[0] / denominator;
#pragma unroll
    for (int edge = 1; edge < E; ++edge) {
        denominator = safe_projection_denominator(
            diagonal[edge] + regularization
            - off_diagonal[edge - 1] * modified_upper[edge - 1]
        );
        modified_rhs[edge] = (
            right_hand_side[edge]
            - off_diagonal[edge - 1] * modified_rhs[edge - 1]
        ) / denominator;
        if (edge < E - 1) modified_upper[edge] = off_diagonal[edge] / denominator;
    }
    multiplier[E - 1] = modified_rhs[E - 1];
#pragma unroll
    for (int edge = E - 2; edge >= 0; --edge) {
        multiplier[edge] = modified_rhs[edge]
            - modified_upper[edge] * multiplier[edge + 1];
    }
#pragma unroll
    for (int edge = 0; edge < E; ++edge) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            const float correction = multiplier[edge] * direction[edge * 3 + axis];
            velocity[edge * 3 + axis] -= inverse_mass[edge] * correction;
            velocity[(edge + 1) * 3 + axis] += inverse_mass[edge + 1] * correction;
        }
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        velocity[axis] = boundary_velocities[boundary_base + axis];
    }
#pragma unroll
    for (int index = 0; index < N * 3; ++index) {
        output_positions[base + index] = value[index];
        output_velocities[base + index] = velocity[index];
    }
}
"""


SUPPORTED_FIXED_NODE_COUNTS = (11, 21, 31)

DampingBackend = Literal[
    "pcg60_reference",
    "pcg32_experimental",
    "pcg24_experimental",
    "pcg16_experimental",
    "block_banded_direct_experimental",
]
REFERENCE_DAMPING_BACKEND: DampingBackend = "pcg60_reference"
# Selected by the August 2026 numerical/closed-loop regression.  The simulator
# default deliberately remains the reference backend; online callers must opt
# in explicitly so old scientific configurations cannot change silently.
VALIDATED_OPTIMIZED_DAMPING_BACKEND: DampingBackend = "pcg32_experimental"
DAMPING_BACKENDS: tuple[DampingBackend, ...] = (
    REFERENCE_DAMPING_BACKEND,
    "pcg32_experimental",
    "pcg24_experimental",
    "pcg16_experimental",
    "block_banded_direct_experimental",
)
_PCG_BACKEND_ITERATIONS: dict[DampingBackend, int] = {
    "pcg60_reference": 60,
    "pcg32_experimental": 32,
    "pcg24_experimental": 24,
    "pcg16_experimental": 16,
}


def validate_damping_backend(value: str) -> DampingBackend:
    if value not in DAMPING_BACKENDS:
        raise ValueError(f"Unknown DDER damping backend: {value!r}.")
    return value  # type: ignore[return-value]


def _replace_cuda_function(source: str, name: str, replacement: str) -> str:
    signature = f'extern "C" __global__ void {name}('
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    end = -1
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        raise RuntimeError(f"Could not isolate CUDA function {name}.")
    return source[:start] + replacement + source[end:]


def _experimental_pcg_source(iterations: int) -> str:
    name = "fixed_damping_11node_60pcg"
    signature = f'extern "C" __global__ void {name}('
    start = _SOURCE.index(signature)
    end = _SOURCE.index(
        'extern "C" __device__ __forceinline__ float safe_projection_denominator',
        start,
    )
    function = _SOURCE[start:end]
    function = function.replace(
        name, f"fixed_damping_11node_{iterations}pcg", 1
    )
    function = function.replace(
        "constexpr int ITERATIONS = 60;",
        f"constexpr int ITERATIONS = {iterations};",
        1,
    )
    return _replace_cuda_function(_SOURCE, name, function.rstrip())


def _direct_damping_source() -> str:
    """Generate a fixed 30-DOF scalar-banded Cholesky diagnostic backend."""

    name = "fixed_damping_11node_60pcg"
    signature = f'extern "C" __global__ void {name}('
    start = _SOURCE.index(signature)
    end = _SOURCE.index(
        'extern "C" __device__ __forceinline__ float safe_projection_denominator',
        start,
    )
    function = _SOURCE[start:end]
    function = function.replace(
        name, "fixed_damping_11node_block_banded_direct", 1
    )
    function = function.replace(
        "    __shared__ double direction_shared[32];",
        "    __shared__ double direction_shared[32];\n"
        "    __shared__ double cholesky_band[D * 9];\n"
        "    __shared__ double forward_solution[D];\n"
        "    __shared__ double direct_solution[D];\n"
        "    __shared__ int factorization_failed;",
        1,
    )
    solve_start = function.index("    double solution = 0.0;")
    output_start = function.index("    if (lane < 3) {", solve_start)
    direct_solve = r"""    if (lane == 0) {
        constexpr int HALF_BANDWIDTH = 8;
        constexpr int L_STRIDE = HALF_BANDWIDTH + 1;
        factorization_failed = 0;
#pragma unroll
        for (int index = 0; index < D * L_STRIDE; ++index) {
            cholesky_band[index] = 0.0;
        }
#pragma unroll
        for (int row = 0; row < D; ++row) {
            const int first_column = row > HALF_BANDWIDTH
                ? row - HALF_BANDWIDTH : 0;
#pragma unroll
            for (int column = first_column; column <= row; ++column) {
                const int row_node = row / 3;
                const int column_node = column / 3;
                const int node_delta = column_node - row_node;
                double value = 0.0;
                if (node_delta >= -2 && node_delta <= 2) {
                    const int matrix_slot = (node_delta + 2) * 3
                        + column % 3;
                    value = matrix_band[row * BAND + matrix_slot];
                }
                int first_inner = row > HALF_BANDWIDTH
                    ? row - HALF_BANDWIDTH : 0;
                const int column_first = column > HALF_BANDWIDTH
                    ? column - HALF_BANDWIDTH : 0;
                if (column_first > first_inner) first_inner = column_first;
#pragma unroll
                for (int inner = first_inner; inner < column; ++inner) {
                    value -= cholesky_band[
                        row * L_STRIDE + row - inner
                    ] * cholesky_band[
                        column * L_STRIDE + column - inner
                    ];
                }
                if (row == column) {
                    if (!(value > 0.0) || !isfinite(value)) {
                        factorization_failed = 1;
                        cholesky_band[row * L_STRIDE] = 1.0;
                    } else {
                        cholesky_band[row * L_STRIDE] = sqrt(value);
                    }
                } else {
                    const double diagonal = cholesky_band[column * L_STRIDE];
                    if (!(diagonal > 0.0) || !isfinite(diagonal)) {
                        factorization_failed = 1;
                        cholesky_band[row * L_STRIDE + row - column] = 0.0;
                    } else {
                        cholesky_band[row * L_STRIDE + row - column] =
                            value / diagonal;
                    }
                }
            }
        }
        if (!factorization_failed) {
#pragma unroll
            for (int row = 0; row < D; ++row) {
                double value = right_hand_side[row];
                const int first_column = row > HALF_BANDWIDTH
                    ? row - HALF_BANDWIDTH : 0;
#pragma unroll
                for (int column = first_column; column < row; ++column) {
                    value -= cholesky_band[
                        row * L_STRIDE + row - column
                    ] * forward_solution[column];
                }
                forward_solution[row] = value / cholesky_band[row * L_STRIDE];
            }
#pragma unroll
            for (int row = D - 1; row >= 0; --row) {
                double value = forward_solution[row];
                const int last_column = row + HALF_BANDWIDTH < D
                    ? row + HALF_BANDWIDTH : D - 1;
#pragma unroll
                for (int column = row + 1; column <= last_column; ++column) {
                    value -= cholesky_band[
                        column * L_STRIDE + column - row
                    ] * direct_solution[column];
                }
                direct_solution[row] = value / cholesky_band[row * L_STRIDE];
            }
        } else {
#pragma unroll
            for (int row = 0; row < D; ++row) {
                direct_solution[row] = __longlong_as_double(
                    0x7ff8000000000000ULL
                );
            }
        }
    }
    __syncwarp();
    const double solution = lane < D ? direct_solution[lane] : 0.0;

"""
    function = function[:solve_start] + direct_solve + function[output_start:]
    return _replace_cuda_function(_SOURCE, name, function.rstrip())


def _block_threads(node_count: int) -> int:
    degrees_of_freedom = 3 * (node_count - 1)
    if degrees_of_freedom <= 32:
        return 32
    if degrees_of_freedom <= 64:
        return 64
    return 128


def _specialized_mechanics_source(node_count: int) -> str:
    """Generate the same fixed-topology mechanics for a supported mesh.

    The validated 11-node source remains byte-for-byte unchanged. Finer meshes
    need more than one warp because their free velocity system has 60 or 90
    scalar degrees of freedom. They therefore use a block-wide reduction while
    retaining the same Jacobian, damping matrix, fixed PCG recurrence, and
    four-plus-one projection algorithm.
    """

    if node_count not in SUPPORTED_FIXED_NODE_COUNTS[1:]:
        raise ValueError("Specialized mechanics generation supports 21 or 31 nodes.")
    degrees_of_freedom = 3 * (node_count - 1)
    residual_count = 3 * (node_count - 2)
    iterations = 6 * (node_count - 1)
    threads = _block_threads(node_count)
    start = _SOURCE.index(
        'extern "C" __device__ __forceinline__ void skew3'
    )
    source = _SOURCE[start:]
    source = source.replace(
        "fixed_damping_11node_60pcg",
        f"fixed_damping_{node_count}node_{iterations}pcg",
    )
    source = source.replace(
        "fixed_projection_11node_4plus1",
        f"fixed_projection_{node_count}node_4plus1",
    )
    source = source.replace("constexpr int N = 11;", f"constexpr int N = {node_count};")
    source = source.replace(
        "constexpr int E = 10;", f"constexpr int E = {node_count - 1};"
    )
    source = source.replace(
        "constexpr int D = 30;", f"constexpr int D = {degrees_of_freedom};"
    )
    source = source.replace(
        "constexpr int R = 27;", f"constexpr int R = {residual_count};"
    )
    source = source.replace(
        "constexpr int ITERATIONS = 60;",
        f"constexpr int ITERATIONS = {iterations};",
    )
    source = source.replace(
        "__shared__ double direction_shared[32];",
        "__shared__ double direction_shared[D];\n"
        "    __shared__ double reduction_shared[4];",
    )
    source = source.replace("index += 32", f"index += {threads}")
    source = source.replace("__syncwarp();", "__syncthreads();")
    source = source.replace(
        "direction_shared[lane] = direction;",
        "if (lane < D) direction_shared[lane] = direction;",
    )
    source = source.replace("column_node < 10", "column_node < N - 1")
    source = source.replace(
        "warp_sum(residual * preconditioned)",
        "block_sum(residual * preconditioned, reduction_shared)",
    )
    source = source.replace(
        "warp_sum(direction * applied)",
        "block_sum(direction * applied, reduction_shared)",
    )
    block_sum = r"""
extern "C" __device__ __forceinline__ double block_sum(
    double value, double* warp_totals
) {
    const int lane_in_warp = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    if (lane_in_warp == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (int)blockDim.x >> 5;
        double total = lane_in_warp < warp_count ? warp_totals[lane_in_warp] : 0.0;
        total += __shfl_down_sync(0xffffffffu, total, 16);
        total += __shfl_down_sync(0xffffffffu, total, 8);
        total += __shfl_down_sync(0xffffffffu, total, 4);
        total += __shfl_down_sync(0xffffffffu, total, 2);
        total += __shfl_down_sync(0xffffffffu, total, 1);
        if (lane_in_warp == 0) warp_totals[0] = total;
    }
    __syncthreads();
    return warp_totals[0];
}
"""
    return block_sum + source


class _CudaError(RuntimeError):
    pass


class _FixedPcgKernel:
    def __init__(
        self,
        node_count: int = 11,
        backend: DampingBackend = REFERENCE_DAMPING_BACKEND,
    ) -> None:
        if node_count not in SUPPORTED_FIXED_NODE_COUNTS:
            raise ValueError(
                f"Fixed mechanics supports node counts {SUPPORTED_FIXED_NODE_COUNTS}."
            )
        self.node_count = node_count
        self.backend = validate_damping_backend(backend)
        if node_count != 11 and self.backend != REFERENCE_DAMPING_BACKEND:
            raise ValueError("Experimental damping backends are currently 11-node only.")
        self.degrees_of_freedom = 3 * (node_count - 1)
        self.iterations = (
            6 * (node_count - 1)
            if self.backend == REFERENCE_DAMPING_BACKEND
            else _PCG_BACKEND_ITERATIONS.get(self.backend, 0)
        )
        self._mechanics_threads = _block_threads(node_count)
        cuda_path = Path(os.environ.get("CUDA_PATH", ""))
        if not cuda_path.is_dir():
            raise RuntimeError("CUDA_PATH does not identify a CUDA Toolkit installation.")
        bin_path = cuda_path / "bin"
        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(bin_path))
        nvrtc_candidates = sorted(bin_path.glob("nvrtc64_*.dll"))
        if not nvrtc_candidates:
            raise RuntimeError(f"NVRTC was not found below {bin_path}.")
        self._nvrtc = ctypes.WinDLL(str(nvrtc_candidates[-1]))
        self._cuda = ctypes.WinDLL("nvcuda.dll")
        self._configure_signatures()
        self._check_cuda(self._cuda.cuInit(0), "cuInit")
        major, minor = torch.cuda.get_device_capability()
        if node_count != 11:
            source = _specialized_mechanics_source(node_count)
        elif self.backend == REFERENCE_DAMPING_BACKEND:
            # Preserve the validated reference source byte-for-byte.
            source = _SOURCE
        elif self.backend == "block_banded_direct_experimental":
            source = _direct_damping_source()
        else:
            source = _experimental_pcg_source(self.iterations)
        ptx = self._compile_ptx(
            f"--gpu-architecture=compute_{major}{minor}",
            source=source,
            source_name=f"fixed_mechanics_{node_count}node_{self.backend}.cu",
        )
        self._module = ctypes.c_void_p()
        self._check_cuda(
            self._cuda.cuModuleLoadDataEx(
                ctypes.byref(self._module),
                ctypes.c_char_p(ptx),
                0,
                None,
                None,
            ),
            "cuModuleLoadDataEx",
        )
        self._function: ctypes.c_void_p | None = None
        if node_count == 11:
            self._function = ctypes.c_void_p()
            self._check_cuda(
                self._cuda.cuModuleGetFunction(
                    ctypes.byref(self._function), self._module, b"fixed_pcg_30x60"
                ),
                "cuModuleGetFunction",
            )
        if self.backend == "block_banded_direct_experimental":
            damping_name = b"fixed_damping_11node_block_banded_direct"
        else:
            damping_name = (
                f"fixed_damping_{node_count}node_{self.iterations}pcg".encode("ascii")
            )
        projection_name = f"fixed_projection_{node_count}node_4plus1".encode(
            "ascii"
        )
        self._damping_function = ctypes.c_void_p()
        self._check_cuda(
            self._cuda.cuModuleGetFunction(
                ctypes.byref(self._damping_function),
                self._module,
                damping_name,
            ),
            f"cuModuleGetFunction({damping_name.decode('ascii')})",
        )
        self._projection_function = ctypes.c_void_p()
        self._check_cuda(
            self._cuda.cuModuleGetFunction(
                ctypes.byref(self._projection_function),
                self._module,
                projection_name,
            ),
            f"cuModuleGetFunction({projection_name.decode('ascii')})",
        )

    def _configure_signatures(self) -> None:
        self._nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        self._nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        self._nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        self._nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
        self._nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        self._nvrtc.nvrtcGetPTX.restype = ctypes.c_int
        self._nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int
        self._nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        self._cuda.cuInit.restype = ctypes.c_int
        self._cuda.cuModuleLoadDataEx.restype = ctypes.c_int
        self._cuda.cuModuleGetFunction.restype = ctypes.c_int
        self._cuda.cuLaunchKernel.restype = ctypes.c_int
        self._cuda.cuGetErrorString.restype = ctypes.c_int

    def _check_nvrtc(self, code: int, operation: str, program=None) -> None:
        if code == 0:
            return
        detail = self._nvrtc.nvrtcGetErrorString(code).decode("utf-8", "replace")
        if program is not None:
            size = ctypes.c_size_t()
            self._nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            if size.value:
                log = ctypes.create_string_buffer(size.value)
                self._nvrtc.nvrtcGetProgramLog(program, log)
                detail += "\n" + log.value.decode("utf-8", "replace")
        raise _CudaError(f"{operation} failed: {detail}")

    def _check_cuda(self, code: int, operation: str) -> None:
        if code == 0:
            return
        message = ctypes.c_char_p()
        self._cuda.cuGetErrorString(code, ctypes.byref(message))
        detail = message.value.decode("utf-8", "replace") if message.value else str(code)
        raise _CudaError(f"{operation} failed: {detail}")

    def _compile_ptx(
        self, architecture: str, *, source: str, source_name: str
    ) -> bytes:
        program = ctypes.c_void_p()
        self._check_nvrtc(
            self._nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                source.encode("utf-8"),
                source_name.encode("ascii"),
                0,
                None,
                None,
            ),
            "nvrtcCreateProgram",
        )
        try:
            encoded_options = [
                architecture.encode("ascii"),
                b"--std=c++17",
                b"--fmad=true",
            ]
            options = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
            code = self._nvrtc.nvrtcCompileProgram(
                program, len(encoded_options), options
            )
            self._check_nvrtc(code, "nvrtcCompileProgram", program)
            size = ctypes.c_size_t()
            self._check_nvrtc(
                self._nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)),
                "nvrtcGetPTXSize",
            )
            output = ctypes.create_string_buffer(size.value)
            self._check_nvrtc(
                self._nvrtc.nvrtcGetPTX(program, output), "nvrtcGetPTX"
            )
            return bytes(output.raw)
        finally:
            self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def launch(self, system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if self._function is None:
            raise ValueError("The standalone dense PCG operator is 11-node only.")
        if system.device.type != "cuda" or rhs.device != system.device:
            raise ValueError("Fixed PCG inputs must be CUDA tensors on one device.")
        if system.dtype != torch.float64 or rhs.dtype != torch.float64:
            raise ValueError("Fixed PCG preserves the float64 damping arithmetic.")
        if system.ndim != 3 or system.shape[1:] != (30, 30):
            raise ValueError("Fixed PCG requires Bx30x30 systems.")
        if rhs.shape != (system.shape[0], 30):
            raise ValueError("Fixed PCG requires Bx30 right-hand sides.")
        if not system.is_contiguous() or not rhs.is_contiguous():
            raise ValueError("Fixed PCG inputs must be contiguous.")
        output = torch.empty_like(rhs)
        system_pointer = ctypes.c_void_p(system.data_ptr())
        rhs_pointer = ctypes.c_void_p(rhs.data_ptr())
        output_pointer = ctypes.c_void_p(output.data_ptr())
        batch = ctypes.c_int(system.shape[0])
        arguments = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(system_pointer), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(rhs_pointer), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(output_pointer), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(batch), ctypes.c_void_p),
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream(system.device).cuda_stream)
        self._check_cuda(
            self._cuda.cuLaunchKernel(
                self._function,
                system.shape[0], 1, 1,
                32, 1, 1,
                0,
                stream,
                arguments,
                None,
            ),
            "cuLaunchKernel(fixed_pcg_30x60)",
        )
        return output

    def launch_damping(
        self,
        positions: torch.Tensor,
        undamped_velocities: torch.Tensor,
        boundary_velocities: torch.Tensor,
        rest_lengths: torch.Tensor,
        masses: torch.Tensor,
        substep_dt: torch.Tensor,
        damping: torch.Tensor,
    ) -> torch.Tensor:
        batch = positions.shape[0]
        expected = (batch, self.node_count, 3)
        if positions.shape != expected or undamped_velocities.shape != expected:
            raise ValueError(
                f"Fixed damping requires Bx{self.node_count}x3 state tensors."
            )
        if boundary_velocities.shape != (batch, 1, 3):
            raise ValueError("Fixed damping requires one prescribed root velocity.")
        tensors = (
            positions,
            undamped_velocities,
            boundary_velocities,
            rest_lengths,
            masses,
            substep_dt,
            damping,
        )
        if any(value.device.type != "cuda" or value.dtype != torch.float32 for value in tensors):
            raise ValueError("Fixed damping preserves the float32 state/float64 PCG runtime path.")
        if rest_lengths.shape != (self.node_count - 1,) or masses.shape != (
            self.node_count,
        ):
            raise ValueError(
                f"Fixed damping requires the fixed {self.node_count}-node topology."
            )
        if substep_dt.shape != (batch,) or damping.shape != (batch,):
            raise ValueError("Fixed damping batch scalars have invalid shapes.")
        contiguous = tuple(value.contiguous() for value in tensors)
        output = torch.empty_like(positions)
        values = [ctypes.c_void_p(value.data_ptr()) for value in contiguous]
        output_pointer = ctypes.c_void_p(output.data_ptr())
        batch_value = ctypes.c_int(batch)
        argument_values = values + [output_pointer, batch_value]
        arguments = (ctypes.c_void_p * len(argument_values))(*[
            ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
            for value in argument_values
        ])
        stream = ctypes.c_void_p(torch.cuda.current_stream(positions.device).cuda_stream)
        self._check_cuda(
            self._cuda.cuLaunchKernel(
                self._damping_function,
                batch, 1, 1,
                self._mechanics_threads, 1, 1,
                0,
                stream,
                arguments,
                None,
            ),
            f"cuLaunchKernel({self.backend})",
        )
        return output

    def launch_projection(
        self,
        predicted_positions: torch.Tensor,
        previous_positions: torch.Tensor,
        rest_lengths: torch.Tensor,
        masses: torch.Tensor,
        boundary_positions: torch.Tensor,
        boundary_velocities: torch.Tensor,
        substep_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = predicted_positions.shape[0]
        expected = (batch, self.node_count, 3)
        if predicted_positions.shape != expected or previous_positions.shape != expected:
            raise ValueError(
                f"Fixed projection requires Bx{self.node_count}x3 positions."
            )
        if boundary_positions.shape != (batch, 1, 3) or boundary_velocities.shape != (batch, 1, 3):
            raise ValueError("Fixed projection requires one root boundary.")
        tensors = (
            predicted_positions,
            previous_positions,
            rest_lengths,
            masses,
            boundary_positions,
            boundary_velocities,
            substep_dt,
        )
        if any(value.device.type != "cuda" or value.dtype != torch.float32 for value in tensors):
            raise ValueError("Fixed projection requires float32 CUDA tensors.")
        if (
            rest_lengths.shape != (self.node_count - 1,)
            or masses.shape != (self.node_count,)
            or substep_dt.shape != (batch,)
        ):
            raise ValueError("Fixed projection topology/scalars are invalid.")
        contiguous = tuple(value.contiguous() for value in tensors)
        output_positions = torch.empty_like(predicted_positions)
        output_velocities = torch.empty_like(predicted_positions)
        values = [ctypes.c_void_p(value.data_ptr()) for value in contiguous]
        q_pointer = ctypes.c_void_p(output_positions.data_ptr())
        v_pointer = ctypes.c_void_p(output_velocities.data_ptr())
        batch_value = ctypes.c_int(batch)
        argument_values = values + [q_pointer, v_pointer, batch_value]
        arguments = (ctypes.c_void_p * len(argument_values))(*[
            ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
            for value in argument_values
        ])
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(predicted_positions.device).cuda_stream
        )
        self._check_cuda(
            self._cuda.cuLaunchKernel(
                self._projection_function,
                batch, 1, 1,
                32, 1, 1,
                0,
                stream,
                arguments,
                None,
            ),
            f"cuLaunchKernel(fixed_projection_{self.node_count}node_4plus1)",
        )
        return output_positions, output_velocities


_lock = threading.Lock()
_kernels: dict[tuple[int, DampingBackend], _FixedPcgKernel] = {}


def _kernel_for(
    node_count: int,
    backend: DampingBackend = REFERENCE_DAMPING_BACKEND,
) -> _FixedPcgKernel:
    if node_count not in SUPPORTED_FIXED_NODE_COUNTS:
        raise ValueError(
            f"Fixed mechanics supports node counts {SUPPORTED_FIXED_NODE_COUNTS}."
        )
    backend = validate_damping_backend(backend)
    key = (node_count, backend)
    kernel = _kernels.get(key)
    if kernel is None:
        with _lock:
            kernel = _kernels.get(key)
            if kernel is None:
                kernel = _FixedPcgKernel(node_count, backend)
                _kernels[key] = kernel
    return kernel


def fixed_pcg_30x60(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve B independent 30-DOF systems with the fixed reference recurrence."""

    return _kernel_for(11).launch(system, rhs)


def fixed_damping_supported_nodes(
    positions: torch.Tensor,
    undamped_velocities: torch.Tensor,
    boundary_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    substep_dt: torch.Tensor,
    damping: torch.Tensor,
    *,
    backend: DampingBackend = REFERENCE_DAMPING_BACKEND,
) -> torch.Tensor:
    """Apply the specialized homogeneous-cable damping for 11/21/31 nodes."""

    return _kernel_for(int(positions.shape[1]), backend).launch_damping(
        positions,
        undamped_velocities,
        boundary_velocities,
        rest_lengths,
        masses,
        substep_dt,
        damping,
    )


def fixed_damping_11node_60pcg(
    positions: torch.Tensor,
    undamped_velocities: torch.Tensor,
    boundary_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    substep_dt: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """Apply exact one-attached 11-node damping in one cooperative kernel."""

    return fixed_damping_supported_nodes(
        positions,
        undamped_velocities,
        boundary_velocities,
        rest_lengths,
        masses,
        substep_dt,
        damping,
    )


def fixed_projection_11node_4plus1(
    predicted_positions: torch.Tensor,
    previous_positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    boundary_positions: torch.Tensor,
    boundary_velocities: torch.Tensor,
    substep_dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply four position and one velocity projection in one CUDA kernel."""

    return fixed_projection_supported_nodes(
        predicted_positions,
        previous_positions,
        rest_lengths,
        masses,
        boundary_positions,
        boundary_velocities,
        substep_dt,
    )


def fixed_projection_supported_nodes(
    predicted_positions: torch.Tensor,
    previous_positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    boundary_positions: torch.Tensor,
    boundary_velocities: torch.Tensor,
    substep_dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply four position plus one velocity projection for 11/21/31 nodes."""

    return _kernel_for(int(predicted_positions.shape[1])).launch_projection(
        predicted_positions,
        previous_positions,
        rest_lengths,
        masses,
        boundary_positions,
        boundary_velocities,
        substep_dt,
    )


def is_available() -> bool:
    return torch.cuda.is_available() and bool(os.environ.get("CUDA_PATH"))
