import numpy as np
import pyopencl as cl
import time


rng = np.random.default_rng()
#phase_steps = rng.random((921600, 64, 2), dtype=np.float32)
#data_fft = rng.random((512, 64, 2), dtype=np.float32)
phase_steps = rng.random((921600 * 64 * 2), dtype=np.float32)
phase_steps /= np.linalg.norm(phase_steps)
data_fft = rng.random((1024 * 64 * 2), dtype=np.float32)
strengths = np.empty((921600,), dtype=np.float32)

print(phase_steps)
print(phase_steps.shape)
print(phase_steps.dtype)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

phase_steps_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=phase_steps)
data_fft_buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_fft)
strengths_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, strengths.nbytes)


prg = cl.Program(ctx, """
float2 complex_multiply(float2 a, float2 b) {
    return (float2)((a.x * b.x - a.y * b.y), (a.x * b.y + a.y * b.x));
}
//float2 complex_multiply(float2 a, float2 b)
//{
//    float k1 = a.x * (b.x + b.y);
//    float k2 = b.y * (a.x + a.y);
//    float k3 = b.x * (a.y - a.x);
//
//    return (float2)(k1 - k2, k1 + k3);
//}

float2 complex_add(float2 a, float2 b) {
    return (float2)((a.x + b.x), (a.y + b.y));
}

__kernel void opencl_kernel_psm(
    __global const float* phase_steps, 
    __global const float* data_fft, 
    __global float *strengths
) {
    const uint P = 921600;
    const uint M = 64;
    const uint S = 1024;

    float4 phases_and_local_phase_steps[M];
    for (uint m = 0; m < M; m++) {
        float phase_step_r = phase_steps[2 * (m * P + get_global_id(0)) + 0];
        float phase_step_i = phase_steps[2 * (m * P + get_global_id(0)) + 1];
        phases_and_local_phase_steps[m] = (float4)(1.0, 0.0, phase_step_r, phase_step_i);
    }

    float strength = 0.0;

    for (uint s = 0; s < S; s++) {
        float2 avg = (float2)(0.0, 0.0);
        for (uint m = 0; m <M; m++) {
            float2 current_phase = (float2)(phases_and_local_phase_steps[m].x, phases_and_local_phase_steps[m].y);
            //float data_fft_r = data_fft[2 * (s * M + m) + 0];
            //float data_fft_i = data_fft[2 * (s * M + m) + 1];
            float data_fft_r = data_fft[2 * (m * S + s) + 0];
            float data_fft_i = data_fft[2 * (m * S + s) + 1];
            avg = complex_add(avg, complex_multiply((float2)(data_fft_r, data_fft_i), current_phase));
            float2 current_phase_step = (float2)(phases_and_local_phase_steps[m].z, phases_and_local_phase_steps[m].w);
            float2 next_phase_step = complex_multiply(current_phase, current_phase_step);
            phases_and_local_phase_steps[m].x = next_phase_step.x;
            phases_and_local_phase_steps[m].y = next_phase_step.y;
        }
        strength += avg.x * avg.x + avg.y * avg.y;
    }

    strengths[get_global_id(0)] = strength;
}
""").build()

for _ in range(10):
    t0 = time.time()
    prg.opencl_kernel_psm(
        queue,
        (921600,),
        (64,),
        phase_steps_buffer,
        data_fft_buffer,
        strengths_buffer
    )

    cl.enqueue_copy(queue, strengths, strengths_buffer)
    queue.finish()
    print("time:", round((1000.0 * (time.time() - t0)), 2), "ms")
print(strengths)
print(strengths.shape)
