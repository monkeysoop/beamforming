from camera import create_camera_directions, create_camera_rotation_matrix
from plot_utils import plot_heatmap
import jax

import numpy as np



CAMERA_DIRECTION = jax.numpy.zeros((3))
UP_DIRECTION = jax.numpy.array([0.0, 1.0, 0.0])
CAMERA_POSITON = jax.numpy.zeros((3))

NUMBER_OF_MICROPHONES = 64
MICROPHONE_POSITIONS = jax.numpy.zeros((NUMBER_OF_MICROPHONES, 3))
MICROPHONE_SAMPLE_RATE = 48000
NUMBER_OF_SAMPLES = 10 * MICROPHONE_SAMPLE_RATE
NUMBER_OF_SAMPLES = 1024
INPUPT_LOCATION = ""
SPEED_OF_SOUND = 343.0

CAMERA_RESOLUTION_WIDTH = 4608
CAMERA_RESOLUTION_HEIGHT = 2592
CAMERA_FOV = 75
CAMERA_FOCAL_LENGTH = 0.00474
CAMERA_SENSOR_DIAGONAL = 0.0074
CAMERA_SENSOR_PIXEL_SIZE = CAMERA_SENSOR_DIAGONAL / ((CAMERA_RESOLUTION_WIDTH * CAMERA_RESOLUTION_WIDTH + CAMERA_RESOLUTION_HEIGHT * CAMERA_RESOLUTION_HEIGHT)**0.5)
CAMERA_SENSOR_WIDTH = CAMERA_SENSOR_PIXEL_SIZE * CAMERA_RESOLUTION_WIDTH
CAMERA_SENSOR_HEIGHT = CAMERA_SENSOR_PIXEL_SIZE * CAMERA_RESOLUTION_HEIGHT
CAMERA_FOCAL_X = (CAMERA_FOCAL_LENGTH / CAMERA_SENSOR_WIDTH) * CAMERA_RESOLUTION_WIDTH
CAMERA_FOCAL_Y = (CAMERA_FOCAL_LENGTH / CAMERA_SENSOR_HEIGHT) * CAMERA_RESOLUTION_HEIGHT

camera_directions = create_camera_directions(CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT, CAMERA_FOCAL_X, CAMERA_FOCAL_Y)

camera_rotation_matrix = jax.device_put(create_camera_rotation_matrix(jax.numpy.array([0.0, 0.0, 1.0]), CAMERA_DIRECTION, UP_DIRECTION))

camera_directions = camera_directions @ camera_rotation_matrix.T

relative_microphone_positions = jax.device_put(MICROPHONE_POSITIONS - CAMERA_POSITON)

camera_pixel_microphone_time_offsets = jax.numpy.tensordot(camera_directions, relative_microphone_positions, axes=([2], [1])) / SPEED_OF_SOUND

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

N = 1024

microphone_samples = jax.device_put(np.random.rand(NUMBER_OF_SAMPLES, NUMBER_OF_MICROPHONES))

PIXEL_CHUNK_SIZE = 1000

audio_strengths = compute_audio_strengths(microphone_samples, N, MICROPHONE_SAMPLE_RATE, camera_pixel_microphone_time_offsets, PIXEL_CHUNK_SIZE)

plot_heatmap(audio_strengths, 10, 10, "audio strength")
