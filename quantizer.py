from delay import Delay

from litex.gen import LiteXModule, Signal, If
from litex.soc.interconnect.stream import Endpoint, EndpointDescription



class Quantizer(LiteXModule):
    def __init__(self, input_data_width, output_data_width, delay_count):
        self.sink = Endpoint(EndpointDescription([("data", input_data_width, True)]), name="quantizer_input")
        self.source = Endpoint(EndpointDescription([("data", output_data_width, True)]), name="quantizer_output")

        width_differene = (input_data_width - output_data_width)
        if (width_differene <= 0):
            delay = Delay(input_data_width, delay_count, delay_count)
            self.submodules += delay

            self.comb += [
                delay.sink.data.eq(self.sink.data),
                delay.sink.valid.eq(self.sink.valid),
                self.source.data.eq(delay.source.data),
                self.source.valid.eq(delay.source.valid),
            ]
        else:
            raw = Signal(bits_sign=(input_data_width, True))
            rounded = Signal(bits_sign=(input_data_width, True))
            quantized = Signal(bits_sign=(input_data_width, True))
            clamped_lower = Signal(bits_sign=(input_data_width, True))
            clamped_lower_and_upper = Signal(bits_sign=(output_data_width, True))
            clamp_min = -int(2**(output_data_width - 1))
            clamp_max = (int(2**(output_data_width - 1)) - 1)

            delay = Delay(output_data_width, max((delay_count - 4), 0), delay_count)
            self.submodules += delay

            self.comb += [
                raw.eq(self.sink.data),

                delay.sink.data.eq(clamped_lower_and_upper),
                self.source.data.eq(delay.source.data),

                delay.sink.valid.eq(self.sink.valid),
                self.source.valid.eq(delay.source.valid),
            ]

            rounder = rounded.eq(raw + (1 << (width_differene - 1)))
            quantizer = quantized.eq(rounded >> width_differene)
            lower_clamper = If((quantized < clamp_min), clamped_lower.eq(clamp_min),).Else(clamped_lower.eq(quantized),)
            upper_clamper = If((clamped_lower > clamp_max), clamped_lower_and_upper.eq(clamp_max),).Else(clamped_lower_and_upper.eq(clamped_lower),)

            if (delay_count == 0):
                self.comb += [
                    rounder,
                    quantizer,
                    lower_clamper,
                    upper_clamper,
                ]
            elif (delay_count == 1):
                self.sync += [
                    rounder,
                ]
                self.comb += [
                    quantizer,
                    lower_clamper,
                    upper_clamper,
                ]
            elif (delay_count == 2):
                self.comb += [
                    rounder,
                    quantizer,
                ]
                self.sync += [
                    lower_clamper,
                    upper_clamper,
                ]
            elif (delay_count == 3):
                self.sync += [
                    rounder,
                ]
                self.comb += [
                    quantizer,
                ]
                self.sync += [
                    lower_clamper,
                    upper_clamper,
                ]
            elif (delay_count >= 4):
                self.sync += [
                    rounder,
                    quantizer,
                    lower_clamper,
                    upper_clamper,
                ]
