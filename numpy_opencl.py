import numpy as np
import pyopencl as cl
import time


rng = np.random.default_rng()
camera_directions = rng.random((921600 * 3), dtype=np.float32)
microphone_positions = rng.random((64 * 3), dtype=np.float32)
data_fft = rng.random((1024 * 64 * 2), dtype=np.float32)
strengths = np.empty((921600,), dtype=np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

camera_directions_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_directions)
microphone_positions_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=microphone_positions)
data_fft_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_fft)
strengths_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, strengths.nbytes)

prg = cl.Program(ctx, """
float2 complex_multiply(float2 a, float2 b) {
    return (float2)((a.x * b.x - a.y * b.y), (a.x * b.y + a.y * b.x));
}

float2 complex_add(float2 a, float2 b) {
    return (float2)((a.x + b.x), (a.y + b.y));
}

__kernel void opencl_kernel_psm(
    __global const float* camera_directions, 
    __global const float* microphone_positions, 
    __global const float* data_fft, 
    __global float* strengths
) {
    const uint NUMBER_OF_MICROPHONE_CHUNKS = 4;
    const uint MICROPHONE_CHUNK_SIZE = 16;
    const uint NUMBER_OF_MICROPHONES = NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE;
    const uint NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS = 64;
    const uint MICROPHONE_SAMPLE_CHUNK_SIZE = 16;
    const uint NUMBER_OF_SAMPLES = NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS * MICROPHONE_SAMPLE_CHUNK_SIZE;
    const float MICROPHONE_SAMPLE_RATE = 48000.0;

    __local float2 shared_shift_steps[NUMBER_OF_MICROPHONES];
    __local float shared_phases[NUMBER_OF_MICROPHONES];

    float3 camera_direction = (float3)(camera_directions[3 * get_global_id(0) + 0], camera_directions[3 * get_global_id(0) + 1], camera_directions[3 * get_global_id(0) + 2]);
    for (uint microphone_index = get_local_id(1); microphone_index < NUMBER_OF_MICROPHONES; microphone_index += get_local_size(1)) {
        float3 microphone_position = (float3)(microphone_positions[3 * microphone_index + 0], microphone_positions[3 * microphone_index + 1], microphone_positions[3 * microphone_index + 2]);
        float phase = -2.0 * M_PI_F * dot(camera_direction, microphone_position) * NUMBER_OF_SAMPLES / MICROPHONE_SAMPLE_RATE;
        shared_phases[microphone_index] = phase;
        float shift_imaginary;
        float shift_real = sincos(phase, &shift_imaginary);
        shared_shift_steps[microphone_index] = (float2)(shift_real, shift_imaginary);
    }

    float2 avgs[MICROPHONE_SAMPLE_CHUNK_SIZE];
    for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
        avgs[microphone_sample_local_index] = (float2)(0.0, 0.0);
    }

    __local float2 shared_data_fft[MICROPHONE_CHUNK_SIZE * (MICROPHONE_SAMPLE_CHUNK_SIZE + 1)];

    for (uint microphone_chunk_index = 0; microphone_chunk_index < NUMBER_OF_MICROPHONE_CHUNKS; microphone_chunk_index++) {
        for (uint i = get_local_id(1); i < (MICROPHONE_CHUNK_SIZE * (MICROPHONE_SAMPLE_CHUNK_SIZE + 1)); i += get_local_size(1)) {
            shared_data_fft[i] = (float2)(data_fft[2 * (microphone_chunk_index * (MICROPHONE_CHUNK_SIZE * (MICROPHONE_SAMPLE_CHUNK_SIZE + 1)) + i) + 0], data_fft[2 * (microphone_chunk_index * (MICROPHONE_CHUNK_SIZE * (MICROPHONE_SAMPLE_CHUNK_SIZE + 1)) + i) + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint microphone_index = 0; microphone_index < MICROPHONE_CHUNK_SIZE; microphone_index++) {
            float phase = get_local_id(1) * shared_phases[microphone_index];
            float shift_imaginary;
            float shift_real = sincos(phase, &shift_imaginary);
            float2 shift = (float2)(shift_real, shift_imaginary);
            for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
                avgs[microphone_sample_local_index] += complex_multiply(shared_data_fft[microphone_index * (MICROPHONE_SAMPLE_CHUNK_SIZE + 1) + microphone_sample_local_index], shift);
                shift = complex_multiply(shift, shared_shift_steps[microphone_index]);
            }
        }
    }

    float strength_local = 0.0;
    for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
        strength_local += avgs[microphone_sample_local_index].x * avgs[microphone_sample_local_index].x + avgs[microphone_sample_local_index].y * avgs[microphone_sample_local_index].y;
    }

    float strength = work_group_reduce_add(strength_local);

    if (get_local_id(1) == 0) {
        strengths[get_global_id(0)] = strength;
    }
}
""").build(options=[ "-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros", "-cl-finite-math-only", ])

for _ in range(10):
    t0 = time.time()
    prg.opencl_kernel_psm(
        queue,
        (921600, 64,),
        (1, 64,),
        camera_directions_buffer,
        microphone_positions_buffer,
        data_fft_buffer,
        strengths_buffer
    )

    cl.enqueue_copy(queue, strengths, strengths_buffer)
    queue.finish()
    print("time:", round((1000.0 * (time.time() - t0)), 2), "ms")
