inline float2 complex_multiply(float2 a, float2 b) {
    return (float2)((a.x * b.x - a.y * b.y), (a.x * b.y + a.y * b.x));
}

__kernel void kernel_beamformer(
    __global const float3* restrict camera_directions,
    __global const float3* restrict microphone_positions,
    __global const float2* restrict data_fft,
    __global float* restrict strength_locals
) {
    const uint NUMBER_OF_MICROPHONES = NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE;
    const uint NUMBER_OF_SAMPLES = NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS * MICROPHONE_SAMPLE_CHUNK_SIZE;

    __local float3 shared_microphone_positions[NUMBER_OF_MICROPHONES];

    for (uint microphone_index = get_local_id(1); microphone_index < NUMBER_OF_MICROPHONES; microphone_index += get_local_size(1)) {
        shared_microphone_positions[microphone_index] = microphone_positions[microphone_index];
    }

    float2 avgs[MICROPHONE_SAMPLE_CHUNK_SIZE];
    for (uint microphone_sample_local_index = 0; microphone_sample_local_index < MICROPHONE_SAMPLE_CHUNK_SIZE; microphone_sample_local_index++) {
        avgs[microphone_sample_local_index] = (float2)(0.0, 0.0);
    }

    __local float2 shared_data_fft[MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE];

    for (uint microphone_chunk_index = 0; microphone_chunk_index < NUMBER_OF_MICROPHONE_CHUNKS; microphone_chunk_index++) {
        for (uint i = get_local_id(1); i < (MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE); i += get_local_size(1)) {
            shared_data_fft[i] = data_fft[get_global_id(0) * (NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE) + microphone_chunk_index * (MICROPHONE_CHUNK_SIZE * MICROPHONE_SAMPLE_CHUNK_SIZE) + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float3 camera_direction = camera_directions[get_global_id(1)];
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

    // correct DC bias term
    if (get_global_id(0) == 0) {
        strength_local -= 0.5 * avgs[0].x * avgs[0].x + avgs[0].y * avgs[0].y;
    }

    strength_locals[get_global_id(1) * NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS + get_global_id(0)] = strength_local;
}

__kernel void kernel_beamformer_reduce(
    __global const float* restrict strength_locals,
    __global float* restrict strengths
) {
    const uint NUMBER_OF_MICROPHONES = NUMBER_OF_MICROPHONE_CHUNKS * MICROPHONE_CHUNK_SIZE;
    const uint NUMBER_OF_SAMPLES = NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS * MICROPHONE_SAMPLE_CHUNK_SIZE;

    __local float shared_strength_locals[NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS];

    shared_strength_locals[get_local_id(0)] = strength_locals[get_global_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = (uint)(get_local_size(0) / 2); stride > 0; stride /= 2) {
        if (get_local_id(0) < stride) {
            shared_strength_locals[get_local_id(0)] += shared_strength_locals[get_local_id(0) + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        strengths[get_group_id(0)] = sqrt(2.0 * shared_strength_locals[0] / NUMBER_OF_MICROPHONES) / NUMBER_OF_SAMPLES;
    }
}
