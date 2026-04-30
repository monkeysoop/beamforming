import jax



@jax.jit
def compute_audio_strengths_for_a_chunk(camera_pixel_microphone_time_offsets, data_fft, data_frequencies, n):
    number_of_microphones = camera_pixel_microphone_time_offsets.shape[1]

    data_shift = jax.numpy.exp(-2.0j * jax.numpy.pi * camera_pixel_microphone_time_offsets[:, jax.numpy.newaxis, :] * data_frequencies[jax.numpy.newaxis, :, jax.numpy.newaxis])
    data_fft_shifted_summed = jax.numpy.einsum("sm,psm->ps", data_fft, data_shift)
    audio_powers = jax.numpy.square(jax.numpy.abs(data_fft_shifted_summed))
    audio_powers = audio_powers.at[:, 1:-1].multiply(2.0)
    audio_strengths = jax.numpy.sqrt(jax.numpy.sum(audio_powers, axis=1)) / n

    return audio_strengths

def compute_audio_strengths(microphone_samples, n, microphone_sample_rate, camera_pixel_microphone_time_offsets, pixel_chunk_size):
    data_fft = jax.numpy.fft.rfft(microphone_samples, axis=0)
    data_frequencies = jax.numpy.fft.rfftfreq(n, d=(1.0 / microphone_sample_rate))

    original_shape = camera_pixel_microphone_time_offsets.shape
    number_of_pixels = original_shape[0] * original_shape[1]
    camera_pixel_microphone_time_offsets = jax.numpy.reshape(camera_pixel_microphone_time_offsets, (number_of_pixels, original_shape[2]))

    delimiters = list(range(0, number_of_pixels, pixel_chunk_size))
    sizes = sizes = [pixel_chunk_size] * (number_of_pixels // pixel_chunk_size)
    if ((number_of_pixels % pixel_chunk_size) != 0):
        sizes.append(number_of_pixels % pixel_chunk_size)

    audio_strengths = jax.device_put(jax.numpy.zeros((number_of_pixels)))
    for delimiter, size in zip(delimiters, sizes):
        audio_strengths = audio_strengths.at[delimiter:(delimiter + size)].set(compute_audio_strengths_for_a_chunk(camera_pixel_microphone_time_offsets[delimiter:(delimiter + size), :], data_fft, data_frequencies, n))

    audio_strengths = jax.numpy.reshape(audio_strengths, (original_shape[0], original_shape[1]))

    return 20.0 * jax.numpy.log10(audio_strengths + 1e-12)
