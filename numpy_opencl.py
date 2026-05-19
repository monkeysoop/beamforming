import numpy as np
import pyopencl as cl
import time



NUMBER_OF_PIXEL_CHUNKS = 14400
PIXEL_CHUNK_SIZE = 64
NUMBER_OF_MICROPHONE_CHUNKS = 4
MICROPHONE_CHUNK_SIZE = 16
NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS = 64
MICROPHONE_SAMPLE_CHUNK_SIZE = 16
MICROPHONE_SAMPLE_RATE = 48000.0

NUMBER_OF_PIXELS = NUMBER_OF_PIXEL_CHUNKS * PIXEL_CHUNK_SIZE
NUMBER_OF_MICROPHONES = NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE
NUMBER_OF_SAMPLES = NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS * MICROPHONE_SAMPLE_CHUNK_SIZE

rng = np.random.default_rng()
camera_directions = rng.random((NUMBER_OF_PIXELS * 3), dtype=np.float32) - 0.5
microphone_positions = rng.random((NUMBER_OF_MICROPHONES * 3), dtype=np.float32)
data_fft = rng.random((NUMBER_OF_SAMPLES * NUMBER_OF_MICROPHONES * 2), dtype=np.float32)
strength_locals = np.empty((NUMBER_OF_PIXELS * NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS), dtype=np.float32)
strengths = np.empty((NUMBER_OF_PIXELS), dtype=np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

camera_directions_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_directions)
microphone_positions_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=microphone_positions)
data_fft_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_fft)
strength_locals_buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, strength_locals.nbytes)
strengths_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, strengths.nbytes)

prg = cl.Program(ctx, """
inline float2 complex_multiply(float2 a, float2 b) {
    return (float2)((a.x * b.x - a.y * b.y), (a.x * b.y + a.y * b.x));
}

__kernel void opencl_kernel_s_p_pp_m_mm_ss(
    __global const float* restrict camera_directions,
    __global const float* restrict microphone_positions,
    __global const float* restrict data_fft,
    __global float* restrict strength_locals
) {
    const uint NUMBER_OF_MICROPHONES = NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE;
    const uint NUMBER_OF_SAMPLES = NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS * MICROPHONE_SAMPLE_CHUNK_SIZE;

    __local float3 shared_microphone_positions[NUMBER_OF_MICROPHONES];

    for (uint microphone_index = get_local_id(1); microphone_index < NUMBER_OF_MICROPHONES; microphone_index += get_local_size(1)) {
        shared_microphone_positions[microphone_index] = (float3)(microphone_positions[3 * microphone_index + 0], microphone_positions[3 * microphone_index + 1], microphone_positions[3 * microphone_index + 2]);
    }

    float2 avgs[MICROPHONE_SAMPLE_CHUNK_SIZE];
    for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
        avgs[microphone_sample_local_index] = (float2)(0.0, 0.0);
    }

    __local float2 shared_data_fft[MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE];

    for (uint microphone_chunk_index = 0; microphone_chunk_index < NUMBER_OF_MICROPHONE_CHUNKS; microphone_chunk_index++) {
        for (uint i = get_local_id(1); i < (MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE); i += get_local_size(1)) {
            uint index = get_global_id(0) * (NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE) + microphone_chunk_index * (MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE) + i;
            shared_data_fft[i] = (float2)(data_fft[2 * (index) + 0], data_fft[2 * (index) + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float3 camera_direction = (float3)(camera_directions[3 * get_global_id(1) + 0], camera_directions[3 * get_global_id(1) + 1], camera_directions[3 * get_global_id(1) + 2]);
        for (uint microphone_local_index = 0; microphone_local_index < MICROPHONE_CHUNK_SIZE; microphone_local_index++) {
            float phase_step = -2.0 * M_PI_F * dot(camera_direction, shared_microphone_positions[microphone_chunk_index * MICROPHONE_CHUNK_SIZE + microphone_local_index]) * MICROPHONE_SAMPLE_RATE / (2 * NUMBER_OF_SAMPLES - 1);
            float shift_step_real;
            float shift_step_imaginary = sincos(phase_step, &shift_step_real);
            float2 shift_step = (float2)(shift_step_real, shift_step_imaginary);
            float shift_real;
            float shift_imaginary = sincos(get_global_id(0) * MICROPHONE_SAMPLE_CHUNK_SIZE * phase_step, &shift_real);
            float2 shift = (float2)(shift_real, shift_imaginary);
            for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
                avgs[microphone_sample_local_index] += complex_multiply(shared_data_fft[microphone_local_index * MICROPHONE_SAMPLE_CHUNK_SIZE + microphone_sample_local_index], shift);
                shift = complex_multiply(shift, shift_step);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float strength_local = 0.0;
    for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
        strength_local += avgs[microphone_sample_local_index].x * avgs[microphone_sample_local_index].x + avgs[microphone_sample_local_index].y * avgs[microphone_sample_local_index].y;
    }

    strength_locals[get_global_id(1) * NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS + get_global_id(0)] = strength_local;
}

__kernel void opencl_kernel_s_p_pp_m_mm_ss_reduce(
    __global const float* restrict strength_locals,
    __global float* restrict strengths
) {
    float strength_local = strength_locals[get_global_id(0)];
    float strength = work_group_reduce_add(strength_local);
    if (get_local_id(0) == 0) {
        strengths[get_group_id(0)] = strength;
    }
}
""")

try:
    prg = prg.build(
        options=[ 
            "-cl-fast-relaxed-math",
            "-cl-mad-enable",
            "-cl-no-signed-zeros",
            "-cl-finite-math-only",
            "-cl-nv-verbose",
            f"-D NUMBER_OF_MICROPHONE_CHUNKS={NUMBER_OF_MICROPHONE_CHUNKS}",
            f"-D MICROPHONE_CHUNK_SIZE={MICROPHONE_CHUNK_SIZE}",
            f"-D NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS={NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS}",
            f"-D MICROPHONE_SAMPLE_CHUNK_SIZE={MICROPHONE_SAMPLE_CHUNK_SIZE}",
            f"-D MICROPHONE_SAMPLE_RATE={MICROPHONE_SAMPLE_RATE}"
        ]
    )
except Exception as e:
    print(e)
    exit(1)


s_p_pp_m_mm_ss_global_sizes = (NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS, (NUMBER_OF_PIXEL_CHUNKS * PIXEL_CHUNK_SIZE),)
s_p_pp_m_mm_ss_local_sizes = (1, PIXEL_CHUNK_SIZE,)

s_p_pp_m_mm_ss_reduce_global_sizes = ((NUMBER_OF_PIXELS * NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS),)
s_p_pp_m_mm_ss_reduce_local_sizes = (NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS,)

for _ in range(10):
    t0 = time.time()
    event_s_p_pp_m_mm_ss = prg.opencl_kernel_s_p_pp_m_mm_ss(
        queue,
        s_p_pp_m_mm_ss_global_sizes,
        s_p_pp_m_mm_ss_local_sizes,
        camera_directions_buffer,
        microphone_positions_buffer,
        data_fft_buffer,
        strength_locals_buffer
    )

    prg.opencl_kernel_s_p_pp_m_mm_ss_reduce(
        queue,
        s_p_pp_m_mm_ss_reduce_global_sizes,
        s_p_pp_m_mm_ss_reduce_local_sizes,
        strength_locals_buffer,
        strengths_buffer,
        wait_for=[event_s_p_pp_m_mm_ss]
    )

    cl.enqueue_copy(queue, strengths, strengths_buffer)
    queue.finish()
    print("time:", round((1000.0 * (time.time() - t0)), 2), "ms")
