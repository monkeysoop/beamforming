from litex.gen import LiteXModule, Signal, If



class PDMCapturerMono(LiteXModule):
    def __init__(
        self,
        oversample_ratio,
        sample_first_index,
        sample_last_index,
        sample_threshold,
        clock_low_first_index,
        clock_high_first_index,
        valid_index,
    ):
        sample_first_index = sample_first_index % oversample_ratio
        sample_last_index = sample_last_index % oversample_ratio
        clock_low_first_index = clock_low_first_index % oversample_ratio
        clock_high_first_index = clock_high_first_index % oversample_ratio
        valid_index = valid_index % oversample_ratio

        number_of_samples = ((sample_last_index - sample_first_index) % oversample_ratio) + 1

        self.microphone_clock = Signal()
        self.microphone_data = Signal()

        read_pdm_data = Signal()

        pdm_data_accumulator = Signal(min=0, max=number_of_samples)

        self.pdm_data = Signal()
        self.pdm_data_valid = Signal()

        counter = Signal(min=0, max=oversample_ratio)
        
        self.comb += [
            self.pdm_data.eq(pdm_data_accumulator >= sample_threshold),
        ]

        self.sync += [
            If((counter == clock_low_first_index),
                self.microphone_clock.eq(0),
            ).Elif((counter == clock_high_first_index),
                self.microphone_clock.eq(1),
            ),

            If((counter == sample_first_index),
                read_pdm_data.eq(1),
                pdm_data_accumulator.eq(self.microphone_data),
            ).Elif((counter == sample_last_index),
                read_pdm_data.eq(0),
            ),

            If((read_pdm_data),
                pdm_data_accumulator.eq(pdm_data_accumulator + self.microphone_data),
            ),

            If((counter == valid_index),
                self.pdm_data_valid.eq(1),
            ).Else(
                self.pdm_data_valid.eq(0),
            ),

            If((counter == (oversample_ratio - 1)),
                counter.eq(0),
            ).Else(
                counter.eq(counter + 1),
            ),
        ]
    
class PDMCapturerStereo(LiteXModule):
    def __init__(
        self,
        oversample_ratio,
        left_sample_first_index,
        left_sample_last_index,
        left_sample_threshold,
        right_sample_first_index,
        right_sample_last_index,
        right_sample_threshold,
        clock_low_first_index,
        clock_high_first_index,
        valid_index,
    ):
        left_sample_first_index = left_sample_first_index % oversample_ratio
        left_sample_last_index = left_sample_last_index % oversample_ratio
        right_sample_first_index = right_sample_first_index % oversample_ratio
        right_sample_last_index = right_sample_last_index % oversample_ratio
        clock_low_first_index = clock_low_first_index % oversample_ratio
        clock_high_first_index = clock_high_first_index % oversample_ratio
        valid_index = valid_index % oversample_ratio

        left_number_of_samples = ((left_sample_last_index - left_sample_first_index) % oversample_ratio) + 1
        right_number_of_samples = ((right_sample_last_index - right_sample_first_index) % oversample_ratio) + 1

        self.microphone_clock = Signal()
        self.microphone_data = Signal()

        read_pdm_data_left = Signal()
        read_pdm_data_right = Signal()

        pdm_data_accumulator_left = Signal(min=0, max=left_number_of_samples)
        pdm_data_accumulator_right = Signal(min=0, max=right_number_of_samples)

        self.pdm_data_left = Signal()
        self.pdm_data_right = Signal()
        self.pdm_data_valid = Signal()

        counter = Signal(min=0, max=oversample_ratio)
        
        self.comb += [
            self.pdm_data_right.eq(pdm_data_accumulator_right >= right_sample_threshold),
            self.pdm_data_left.eq(pdm_data_accumulator_left >= left_sample_threshold),
        ]

        self.sync += [
            If((counter == clock_low_first_index),
                self.microphone_clock.eq(0),
            ).Elif((counter == clock_high_first_index),
                self.microphone_clock.eq(1),
            ),

            If((counter == left_sample_first_index),
                read_pdm_data_left.eq(1),
                pdm_data_accumulator_left.eq(self.microphone_data),
            ).Elif((counter == left_sample_last_index),
                read_pdm_data_left.eq(0),
            ).Elif((counter == right_sample_first_index),
                read_pdm_data_right.eq(1),
                pdm_data_accumulator_right.eq(self.microphone_data),
            ).Elif((counter == right_sample_last_index),
                read_pdm_data_right.eq(0),
            ),

            If((read_pdm_data_left),
                pdm_data_accumulator_left.eq(pdm_data_accumulator_left + self.microphone_data),
            ).Elif((read_pdm_data_right),
                pdm_data_accumulator_right.eq(pdm_data_accumulator_right + self.microphone_data),
            ),

            If((counter == valid_index),
                self.pdm_data_valid.eq(1),
            ).Else(
                self.pdm_data_valid.eq(0),
            ),

            If((counter == (oversample_ratio - 1)),
                counter.eq(0),
            ).Else(
                counter.eq(counter + 1),
            ),
        ]
