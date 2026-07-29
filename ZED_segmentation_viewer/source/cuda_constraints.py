"""Fused CUDA kernels for fixed-link position and velocity projection."""

from __future__ import annotations

import torch


_CUDA_SOURCE = r"""
extern "C" __global__ void constrain_fixed_links_kernel(
    const float* input,
    const float* endpoints,
    const float* segment_lengths,
    float* output,
    int cable_count,
    int particle_count,
    int node_count,
    int iterations,
    int visibility_bits
) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_particles = cable_count * particle_count;
    if (particle >= total_particles) {
        return;
    }

    const int cable_index = particle / particle_count;
    const int base = particle * node_count * 3;
    for (int index = 0; index < node_count * 3; ++index) {
        output[base + index] = input[base + index];
    }

    const bool start_visible =
        (visibility_bits & (1 << (2 * cable_index))) != 0;
    const bool end_visible =
        (visibility_bits & (1 << (2 * cable_index + 1))) != 0;
    const float segment_length = segment_lengths[cable_index];
    const float epsilon = 1.1920928955078125e-7f;

    #define SET_ENDPOINT(node, endpoint_index) do {                         \
        const int destination = base + (node) * 3;                          \
        const int source = (cable_index * 2 + (endpoint_index)) * 3;        \
        output[destination] = endpoints[source];                            \
        output[destination + 1] = endpoints[source + 1];                    \
        output[destination + 2] = endpoints[source + 2];                    \
    } while (0)

    #define ENFORCE_LINK(destination_node, source_node) do {                 \
        const int destination = base + (destination_node) * 3;              \
        const int source = base + (source_node) * 3;                        \
        const float x = output[destination] - output[source];               \
        const float y = output[destination + 1] - output[source + 1];       \
        const float z = output[destination + 2] - output[source + 2];       \
        const float xx = x * x;                                             \
        const float yy = y * y;                                             \
        const float zz = z * z;                                             \
        const float norm = fmaxf(sqrtf((xx + zz) + yy), epsilon);           \
        const float unit_x = x / norm;                                      \
        const float unit_y = y / norm;                                      \
        const float unit_z = z / norm;                                      \
        const float delta_x = segment_length * unit_x;                      \
        const float delta_y = segment_length * unit_y;                      \
        const float delta_z = segment_length * unit_z;                      \
        output[destination] = output[source] + delta_x;                     \
        output[destination + 1] = output[source + 1] + delta_y;             \
        output[destination + 2] = output[source + 2] + delta_z;             \
    } while (0)

    if (start_visible && end_visible) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            SET_ENDPOINT(node_count - 1, 1);
            for (int node = node_count - 2; node >= 0; --node) {
                ENFORCE_LINK(node, node + 1);
            }
            SET_ENDPOINT(0, 0);
            for (int node = 0; node < node_count - 1; ++node) {
                ENFORCE_LINK(node + 1, node);
            }
        }
    } else if (start_visible) {
        SET_ENDPOINT(0, 0);
        for (int node = 0; node < node_count - 1; ++node) {
            ENFORCE_LINK(node + 1, node);
        }
    } else if (end_visible) {
        SET_ENDPOINT(node_count - 1, 1);
        for (int node = node_count - 2; node >= 0; --node) {
            ENFORCE_LINK(node, node + 1);
        }
    }

    #undef ENFORCE_LINK
    #undef SET_ENDPOINT
}

extern "C" __global__ void project_link_velocities_kernel(
    const float* velocities,
    const float* particles,
    float* output,
    int cable_count,
    int particle_count,
    int node_count,
    int iterations,
    int visibility_bits
) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_particles = cable_count * particle_count;
    if (particle >= total_particles) {
        return;
    }

    const int cable_index = particle / particle_count;
    const int base = particle * node_count * 3;
    for (int index = 0; index < node_count * 3; ++index) {
        output[base + index] = velocities[base + index];
    }

    const bool start_visible =
        (visibility_bits & (1 << (2 * cable_index))) != 0;
    const bool end_visible =
        (visibility_bits & (1 << (2 * cable_index + 1))) != 0;
    const float epsilon = 1.1920928955078125e-7f;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int node = 0; node < node_count - 1; ++node) {
            const int first = base + node * 3;
            const int second = first + 3;
            float x = particles[second] - particles[first];
            float y = particles[second + 1] - particles[first + 1];
            float z = particles[second + 2] - particles[first + 2];
            const float xx = x * x;
            const float yy = y * y;
            const float zz = z * z;
            const float norm = fmaxf(sqrtf((xx + zz) + yy), epsilon);
            x /= norm;
            y /= norm;
            z /= norm;

            const float velocity_x = output[second] - output[first];
            const float velocity_y = output[second + 1] - output[first + 1];
            const float velocity_z = output[second + 2] - output[first + 2];
            const float product_x = velocity_x * x;
            const float product_y = velocity_y * y;
            const float product_z = velocity_z * z;
            const float radial = (product_x + product_z) + product_y;
            const bool first_fixed = node == 0 && start_visible;
            const bool second_fixed =
                node + 1 == node_count - 1 && end_visible;

            if (first_fixed && !second_fixed) {
                output[second] -= radial * x;
                output[second + 1] -= radial * y;
                output[second + 2] -= radial * z;
            } else if (second_fixed && !first_fixed) {
                output[first] += radial * x;
                output[first + 1] += radial * y;
                output[first + 2] += radial * z;
            } else if (!first_fixed && !second_fixed) {
                const float half_radial = 0.5f * radial;
                const float correction_x = half_radial * x;
                const float correction_y = half_radial * y;
                const float correction_z = half_radial * z;
                output[first] += correction_x;
                output[first + 1] += correction_y;
                output[first + 2] += correction_z;
                output[second] -= correction_x;
                output[second + 1] -= correction_y;
                output[second + 2] -= correction_z;
            }
        }
    }
}

"""


class FusedCudaConstraints:
    """Launch one native CUDA thread per independent cable particle."""

    _threads_per_block = 128

    def __init__(self, device: torch.device):
        if device.type != "cuda":
            raise ValueError("FusedCudaConstraints requires a CUDA device")
        if not hasattr(torch.cuda, "_compile_kernel"):
            raise RuntimeError(
                "This PyTorch build does not provide the NVRTC kernel compiler"
            )
        self.device = device
        with torch.cuda.device(device):
            options = ["--fmad=false"]
            self._position_kernel = torch.cuda._compile_kernel(
                _CUDA_SOURCE,
                "constrain_fixed_links_kernel",
                nvcc_options=options,
            )
            self._velocity_kernel = torch.cuda._compile_kernel(
                _CUDA_SOURCE,
                "project_link_velocities_kernel",
                nvcc_options=options,
            )

    @staticmethod
    def _visibility_bits(
        endpoint_visible: tuple[tuple[bool, bool], ...],
    ) -> int:
        bits = 0
        for cable_index, visibility in enumerate(endpoint_visible):
            if len(visibility) != 2:
                raise ValueError("Each cable must provide two endpoint flags")
            for endpoint_index, visible in enumerate(visibility):
                bits |= int(bool(visible)) << (2 * cable_index + endpoint_index)
        return bits

    def _validate_particles(self, tensor: torch.Tensor, name: str) -> None:
        if (
            tensor.device != self.device
            or tensor.dtype != torch.float32
            or tensor.ndim != 4
            or tensor.shape[-1] != 3
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA float32 [cable, particle, node, 3]"
            )

    def constrain(
        self,
        particles: torch.Tensor,
        endpoints: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], ...],
        segment_lengths: torch.Tensor,
        iterations: int,
    ) -> torch.Tensor:
        self._validate_particles(particles, "particles")
        cable_count, particle_count, node_count, _ = particles.shape
        if len(endpoint_visible) != cable_count:
            raise ValueError("Endpoint visibility must match the cable dimension")
        if any(not (visibility[0] or visibility[1]) for visibility in endpoint_visible):
            raise ValueError(
                "Fused position projection requires at least one visible endpoint "
                "per cable"
            )
        if iterations < 1:
            raise ValueError("Position projection requires at least one iteration")
        if (
            endpoints.shape != (cable_count, 2, 3)
            or endpoints.device != self.device
            or endpoints.dtype != torch.float32
            or not endpoints.is_contiguous()
        ):
            raise ValueError("endpoints must be contiguous CUDA float32 [cable, 2, 3]")
        if (
            segment_lengths.shape != (cable_count,)
            or segment_lengths.device != self.device
            or segment_lengths.dtype != torch.float32
            or not segment_lengths.is_contiguous()
        ):
            raise ValueError(
                "segment_lengths must be contiguous CUDA float32 [cable]"
            )
        output = torch.empty_like(particles)
        total_particles = cable_count * particle_count
        block_count = (
            total_particles + self._threads_per_block - 1
        ) // self._threads_per_block
        with torch.cuda.device(self.device):
            self._position_kernel(
                grid=(block_count, 1, 1),
                block=(self._threads_per_block, 1, 1),
                args=[
                    particles,
                    endpoints,
                    segment_lengths,
                    output,
                    cable_count,
                    particle_count,
                    node_count,
                    int(iterations),
                    self._visibility_bits(endpoint_visible),
                ],
            )
        return output
    def project_velocities(
        self,
        velocities: torch.Tensor,
        particles: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], ...],
        iterations: int,
    ) -> torch.Tensor:
        self._validate_particles(velocities, "velocities")
        self._validate_particles(particles, "particles")
        if velocities.shape != particles.shape:
            raise ValueError("velocities and particles must have identical shapes")
        cable_count, particle_count, node_count, _ = particles.shape
        if len(endpoint_visible) != cable_count:
            raise ValueError("Endpoint visibility must match the cable dimension")
        if iterations < 1:
            raise ValueError("Velocity projection requires at least one iteration")
        output = torch.empty_like(velocities)
        total_particles = cable_count * particle_count
        block_count = (
            total_particles + self._threads_per_block - 1
        ) // self._threads_per_block
        with torch.cuda.device(self.device):
            self._velocity_kernel(
                grid=(block_count, 1, 1),
                block=(self._threads_per_block, 1, 1),
                args=[
                    velocities,
                    particles,
                    output,
                    cable_count,
                    particle_count,
                    node_count,
                    int(iterations),
                    self._visibility_bits(endpoint_visible),
                ],
            )
        return output
