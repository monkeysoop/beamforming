from litex.gen import LiteXModule, Signal, If
from litex.soc.interconnect.stream import Endpoint, EndpointDescription



class Quantizer(LiteXModule):
    def __init__(self, input_data_width, output_data_width, delay_count):
        self.sink = Endpoint(EndpointDescription([("data", input_data_width, True)]), name="quantizer_input")
        self.source = Endpoint(EndpointDescription([("data", output_data_width, True)]), name="quantizer_output")

        width_differene = (input_data_width - output_data_width)
        if (width_differene <= 0):
            data_delays = [Signal(bits_sign=(input_data_width, True)) for _ in range(max(delay_count, 0) + 1)]
            valid_delays = [Signal() for _ in range(max(delay_count, 0) + 1)]
            self.comb += [
                data_delays[0].eq(self.sink.data),
                valid_delays[0].eq(self.sink.valid),
                self.source.data.eq(data_delays[-1]),
                self.source.valid.eq(valid_delays[-1]),
            ]

            for i in range(delay_count):
                self.sync += [
                    data_delays[i + 1].eq(data_delays[i]),
                    valid_delays[i + 1].eq(valid_delays[i]),
                ]
        else:
            raw = Signal(bits_sign=(input_data_width, True))
            rounded = Signal(bits_sign=(input_data_width, True))
            quantized = Signal(bits_sign=(input_data_width, True))
            clamped_lower = Signal(bits_sign=(input_data_width, True))
            clamped_lower_and_upper = Signal(bits_sign=(output_data_width, True))
            clamp_min = -int(2**(output_data_width - 1))
            clamp_max = (int(2**(output_data_width - 1)) - 1)


            data_delays = [Signal(bits_sign=(output_data_width, True)) for _ in range(max((delay_count - 4), 0) + 1)]
            valid_delays = [Signal() for _ in range(max(delay_count, 0) + 1)]

            for i in range(1, len(data_delays)):
                self.sync += [
                    data_delays[i].eq(data_delays[i - 1]),
                ]
            for i in range(1, len(valid_delays)):
                self.sync += [
                    valid_delays[i].eq(valid_delays[i - 1]),
                ]

            self.comb += [
                raw.eq(self.sink.data),

                data_delays[0].eq(clamped_lower_and_upper),
                self.source.data.eq(data_delays[-1]),

                valid_delays[0].eq(self.sink.valid),
                self.source.valid.eq(valid_delays[-1]),
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
