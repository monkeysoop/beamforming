from litex.gen import LiteXModule, Signal, If, Array
from litex.soc.interconnect.stream import Endpoint, EndpointDescription, SyncFIFO

from migen.sim import run_simulation

from functools import reduce



class Multiplexer(LiteXModule):
    def __init__(self, number_of_inputs, data_width):
        self.sinks = [Endpoint(EndpointDescription([("data", data_width)]), name="multiplexer_input") for _ in range(number_of_inputs)]
        self.source = Endpoint(EndpointDescription([("data", data_width)]), name="multiplexer_output")

        self.any_buffer_valid = Signal()

        self.data_buffer = Array([Signal(data_width) for _ in range(number_of_inputs)])
        self.valid_buffer = Array([Signal() for _ in range(number_of_inputs)])

        input_counter = Signal(min=0, max=(number_of_inputs + 1))

        self.comb += [
            self.any_buffer_valid.eq(reduce(lambda a, b: (a | b), [self.valid_buffer[i] for i in range(number_of_inputs)])),
        ]

        for i in range(number_of_inputs):
            self.comb += [
                self.sinks[i].ready.eq(~self.valid_buffer[i] | (input_counter == i)),
            ]
            self.sync += [
                If((self.sinks[i].valid),
                    self.data_buffer[i].eq(self.sinks[i].data),
                ),
                self.valid_buffer[i].eq((self.valid_buffer[i] & (input_counter != i)) | self.sinks[i].valid),
            ]

        self.sync += [
            self.source.data.eq(self.data_buffer[input_counter]),
            self.source.valid.eq(self.valid_buffer[input_counter]),
            If((self.any_buffer_valid),
                If((input_counter == (number_of_inputs - 1)),
                    input_counter.eq(0),
                ).Else(
                    input_counter.eq(input_counter + 1),
                ),
            ).Else(
                input_counter.eq(0),
            ),
        ]

class BurstMultiplexer(LiteXModule):
    def __init__(self, number_of_inputs, data_width):
        self.sinks = [Endpoint(EndpointDescription([("data", data_width)]), name="multiplexer_input") for _ in range(number_of_inputs)]
        self.source = Endpoint(EndpointDescription([("data", data_width)]), name="multiplexer_output")

        self.fifos = []
        for _ in range(number_of_inputs):
            fifo = SyncFIFO(
                layout=[("data", data_width)],
                depth=number_of_inputs,
                buffered=True
            )
            self.submodules += fifo
            self.fifos.append(fifo)

        self.multiplexer = Multiplexer(
            number_of_inputs=number_of_inputs,
            data_width=data_width,
        )

        for i in range(number_of_inputs):
            self.sync += [
                self.sinks[i].connect(self.fifos[i].sink),
            ]
            self.comb += [
                self.fifos[i].source.ready.eq(self.fifos[i].source.valid & self.multiplexer.sinks[i].ready),
                self.multiplexer.sinks[i].data.eq(self.fifos[i].source.data),
                self.multiplexer.sinks[i].valid.eq(self.fifos[i].source.valid & self.multiplexer.sinks[i].ready),
            ]

        self.comb += [
            self.multiplexer.source.connect(self.source),
        ]
