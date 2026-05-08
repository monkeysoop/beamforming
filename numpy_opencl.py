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
    __global float *strengths
) {
    const uint P = 921600;
    const uint M = 64;
    const uint S = 64;
    const uint SS = 16;

    __local float2 shared_shift_steps[M];
    __local float shared_phases[M];

    float3 camera_direction = (float3)(camera_directions[3 * get_global_id(0) + 0], camera_directions[3 * get_global_id(0) + 1], camera_directions[3 * get_global_id(0) + 2]);
    for (uint m = get_local_id(1); m < M; m += get_local_size(1)) {
        float3 microphone_position = (float3)(microphone_positions[3 * m + 0], microphone_positions[3 * m + 1], microphone_positions[3 * m + 2]);
        float phase = 2.0 * M_PI_F * dot(camera_direction, microphone_position) * S * SS / 48000.0;
        shared_phases[m] = phase;
        float shift_imaginary;
        float shift_real = sincos(phase, &shift_imaginary);
        shared_shift_steps[m] = (float2)(shift_real, shift_imaginary);
    }
 
    __local float2 shared_data_fft[M * SS];
    for (uint i = get_local_id(1); i < (M * SS); i += get_local_size(1)) {
        shared_data_fft[i] = (float2)(data_fft[2 * i + 0], data_fft[2 * i + 1]);
    }

    float2 avgs[SS];
    for (uint ss = 0; ss < SS; ss++) {
        avgs[ss] = (float2)(0.0, 0.0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint m = 0; m < M; m++) {
        float phase = get_global_id(1) * shared_phases[m];
        float shift_imaginary;
        float shift_real = sincos(phase, &shift_imaginary);
        float2 shift = (float2)(shift_real, shift_imaginary);
        for (uint ss = 0; ss < SS; ss++) {
            avgs[ss] = complex_add(avgs[ss], complex_multiply(shared_data_fft[m * SS + ss], shift));
            shift = complex_multiply(shift, shared_shift_steps[m]);
        }
    }

    float strength_local = 0.0;
    for (uint ss = 0; ss < SS; ss++) {
        strength_local += avgs[SS].x * avgs[SS].x + avgs[SS].y * avgs[SS].y;
    }

    float strength = work_group_reduce_add(strength_local);

    if (get_local_id(1) == 0) {
        strengths[get_group_id(0)] = strength;
    }
}
""").build()

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
