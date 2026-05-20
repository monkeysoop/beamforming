import numpy as np
import pyopencl as cl
import time



class BeamformerSPPPMMMSS:
    OPENCL_KERNEL_FILENAME = "beamformer_s_p_pp_m_mm_ss.cl"

    def __init__(
        self,
        number_of_pixel_chunks,
        pixel_chunk_size,
        number_of_microphone_chunks,
        microphone_chunk_size,
        number_of_microphone_sample_chunks,
        microphone_sample_chunk_size,
        microphone_sample_rate,
        camera_directions,
        microphone_positions,
        data_fft
    ):
        self.number_of_pixel_chunks = number_of_pixel_chunks
        self.pixel_chunk_size = pixel_chunk_size
        self.number_of_microphone_chunks = number_of_microphone_chunks
        self.microphone_chunk_size = microphone_chunk_size
        self.number_of_microphone_sample_chunks = number_of_microphone_sample_chunks
        self.microphone_sample_chunk_size = microphone_sample_chunk_size
        self.microphone_sample_rate = microphone_sample_rate

        self.number_of_pixels = self.number_of_pixel_chunks * self.pixel_chunk_size
        self.number_of_microphones = self.number_of_microphone_chunks * self.microphone_chunk_size
        self.number_of_samples = self.number_of_microphone_sample_chunks * self.microphone_sample_chunk_size

        self.strength_locals = np.empty((self.number_of_pixels * self.number_of_microphone_sample_chunks), dtype=np.float32)
        self.strengths = np.empty((self.number_of_pixels), dtype=np.float32)

        self.ctx = cl.create_some_context()

        self.camera_directions_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_directions)
        self.microphone_positions_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=microphone_positions)
        self.data_fft_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_fft)
        self.strength_locals_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.strength_locals.nbytes)
        self.strengths_buffer = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, self.strengths.nbytes)

        self.prg = None
        with open(self.OPENCL_KERNEL_FILENAME, "r") as opencl_kernel_file:
            self.prg = cl.Program(self.ctx, opencl_kernel_file.read()).build(
                options=[ 
                    "-cl-fast-relaxed-math",
                    "-cl-mad-enable",
                    "-cl-no-signed-zeros",
                    "-cl-finite-math-only",
                    f"-D NUMBER_OF_MICROPHONE_CHUNKS={self.number_of_microphone_chunks}",
                    f"-D MICROPHONE_CHUNK_SIZE={self.microphone_chunk_size}",
                    f"-D NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS={self.number_of_microphone_sample_chunks}",
                    f"-D MICROPHONE_SAMPLE_CHUNK_SIZE={self.microphone_sample_chunk_size}",
                    f"-D MICROPHONE_SAMPLE_RATE={self.microphone_sample_rate}"
                ]
            )

        self.kernel_beamformer_global_sizes = (self.number_of_microphone_sample_chunks, (self.number_of_pixel_chunks * self.pixel_chunk_size),)
        self.kernel_beamformer_local_sizes = (1, self.pixel_chunk_size,)
        self.kernel_beamformer_reduce_global_sizes = ((self.number_of_pixels * self.number_of_microphone_sample_chunks),)
        self.kernel_beamformer_reduce_local_sizes = (self.number_of_microphone_sample_chunks,)

    def update_camera_directions(self, camera_directions):
        self.camera_directions_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_directions)

    def update_microphone_positions(self, microphone_positions):
        self.microphone_positions_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=microphone_positions)

    def update_data_fft(self, data_fft):
        self.data_fft_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_fft)

    def beamform(self):
        queue = cl.CommandQueue(self.ctx)
        kernel_beamformer_event = self.prg.kernel_beamformer(
            queue,
            self.kernel_beamformer_global_sizes,
            self.kernel_beamformer_local_sizes,
            self.camera_directions_buffer,
            self.microphone_positions_buffer,
            self.data_fft_buffer,
            self.strength_locals_buffer
        )

        self.prg.kernel_beamformer_reduce(
            queue,
            self.kernel_beamformer_reduce_global_sizes,
            self.kernel_beamformer_reduce_local_sizes,
            self.strength_locals_buffer,
            self.strengths_buffer,
            wait_for=[kernel_beamformer_event]
        )

        cl.enqueue_copy(queue, self.strengths, self.strengths_buffer)
        queue.finish()

        return self.strengths

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
data_fft = np.fft.rfft((rng.random(((2 * NUMBER_OF_SAMPLES - 1), NUMBER_OF_MICROPHONES), dtype=np.float32) - 0.5), axis=0).reshape(NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS, MICROPHONE_SAMPLE_CHUNK_SIZE, NUMBER_OF_MICROPHONE_CHUNKS, MICROPHONE_CHUNK_SIZE).swapaxes(1, 2).swapaxes(2, 3).reshape(NUMBER_OF_MICROPHONES * NUMBER_OF_SAMPLES).view(np.float32)

beamformer = BeamformerSPPPMMMSS(
    NUMBER_OF_PIXEL_CHUNKS,
    PIXEL_CHUNK_SIZE,
    NUMBER_OF_MICROPHONE_CHUNKS,
    MICROPHONE_CHUNK_SIZE,
    NUMBER_OF_MICROPHONE_SAMPLE_CHUNKS,
    MICROPHONE_SAMPLE_CHUNK_SIZE,
    MICROPHONE_SAMPLE_RATE,
    camera_directions,
    microphone_positions,
    data_fft
)

strengths = None
for _ in range(10):
    t0 = time.time()
    strengths = beamformer.beamform()
    print("time:", round((1000.0 * (time.time() - t0)), 2), "ms")
