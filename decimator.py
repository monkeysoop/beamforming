from litex.gen import LiteXModule, Signal, If
from litex.soc.interconnect.stream import Endpoint, EndpointDescription



class Decimator(LiteXModule):
    def __init__(self, number_of_pipelines, decimation_ratio, data_width):
        self.sink = Endpoint(EndpointDescription([("data", data_width)]), name="decimator_input")
        self.source = Endpoint(EndpointDescription([("data", data_width)]), name="decimator_output")

        pipeline_counter = Signal(min=0, max=number_of_pipelines + 1)
        decimation_counter = Signal(min=0, max=decimation_ratio + 1)

        self.sync += [
            self.source.data.eq(self.sink.data),
            If((self.sink.valid),
                If((decimation_counter == (decimation_ratio - 1)),
                    self.source.valid.eq(1),
                ).Else(
                    self.source.valid.eq(0),
                ),
                If((pipeline_counter == (number_of_pipelines - 1)),
                    pipeline_counter.eq(0),
                    If((decimation_counter == (decimation_ratio - 1)),
                        decimation_counter.eq(0),
                    ).Else(
                        decimation_counter.eq(decimation_counter + 1),
                    ),
                ).Else(
                    pipeline_counter.eq(pipeline_counter + 1),
                ),
            ).Else(
                self.source.valid.eq(0),
            ),
        ]
