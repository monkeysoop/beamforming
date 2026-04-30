from camera import create_camera_directions, create_camera_rotation_matrix
from plot_utils import plot_heatmap
from beamforming import compute_audio_strengths
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

N = 1024

microphone_samples = jax.device_put(np.random.rand(NUMBER_OF_SAMPLES, NUMBER_OF_MICROPHONES))

PIXEL_CHUNK_SIZE = 1000

audio_strengths = compute_audio_strengths(microphone_samples, N, MICROPHONE_SAMPLE_RATE, camera_pixel_microphone_time_offsets, PIXEL_CHUNK_SIZE)

plot_heatmap(audio_strengths, 10, 10, "audio strength")
