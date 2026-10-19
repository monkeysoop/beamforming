from litex.gen import LiteXModule, Signal
from litex.soc.interconnect.stream import Endpoint, EndpointDescription



class Delay(LiteXModule):
    def __init__(self, data_width, data_delay_count, valid_delay_count):
        self.sink = Endpoint(EndpointDescription([("data", data_width)]), name="delay_input")
        self.source = Endpoint(EndpointDescription([("data", data_width)]), name="delay_output")

        data_delays = [Signal(data_width) for _ in range(max(data_delay_count, 0) + 1)]
        valid_delays = [Signal() for _ in range(max(valid_delay_count, 0) + 1)]

        self.comb += [
            data_delays[0].eq(self.sink.data),
            self.source.data.eq(data_delays[-1]),

            valid_delays[0].eq(self.sink.valid),
            self.source.valid.eq(valid_delays[-1]),
        ]

        for i in range(data_delay_count):
            self.sync += [
                data_delays[i + 1].eq(data_delays[i]),
            ]

        for i in range(valid_delay_count):
            self.sync += [
                valid_delays[i + 1].eq(valid_delays[i]),
            ]
