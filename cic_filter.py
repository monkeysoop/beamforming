from decimator import Decimator
from multiplexer import BurstMultiplexer

from litex.gen import LiteXModule, Signal, If, Memory, WRITE_FIRST
from litex.soc.interconnect.stream import Endpoint, EndpointDescription
from migen.sim import run_simulation

import math



class CICFilterIntegratorStage(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, clock_domain):
        assert (number_of_pipelines >= 2), "invalid number of pipelines"

        self.sink = Endpoint(EndpointDescription([("data", data_width, True)]), name="integrator_stage_input")
        self.source = Endpoint(EndpointDescription([("data", data_width, True)]), name="integrator_stage_output")

        integrator_memory = Memory(width=data_width, depth=number_of_pipelines)
        integrator_write_memory_port = integrator_memory.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        integrator_read_memory_port = integrator_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += integrator_memory
        self.specials += integrator_write_memory_port
        self.specials += integrator_read_memory_port

        write_offset = 0 % number_of_pipelines
        read_offset = 2 % number_of_pipelines

        write_counter = Signal(min=0, max=number_of_pipelines, reset=write_offset)
        read_counter = Signal(min=0, max=number_of_pipelines, reset=read_offset)

        self.sync += [
            integrator_write_memory_port.we.eq(self.sink.valid),
            integrator_write_memory_port.adr.eq(write_counter),

            integrator_read_memory_port.re.eq(self.sink.valid),
            integrator_read_memory_port.adr.eq(read_counter),
        ]

        self.sync += [
            If((self.sink.valid),
                If((write_counter == (number_of_pipelines - 1)),
                    write_counter.eq(0),
                ).Else(
                    write_counter.eq(write_counter + 1),
                ),
                If((read_counter == (number_of_pipelines - 1)),
                    read_counter.eq(0),
                ).Else(
                    read_counter.eq(read_counter + 1),
                ),
            ),
        ]

        valid_delay = Signal()

        self.sync += [
            valid_delay.eq(self.sink.valid),
            If((self.sink.valid),
                integrator_write_memory_port.dat_w.eq(integrator_read_memory_port.dat_r + self.sink.data),
            ),
        ]

        self.sync += [
            self.source.data.eq(integrator_write_memory_port.dat_w),
            self.source.valid.eq(valid_delay),
        ]

class CICFilterCombStage(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, clock_domain):
        self.sink = Endpoint(EndpointDescription([("data", data_width, True)]), name="comb_stage_input")
        self.source = Endpoint(EndpointDescription([("data", data_width, True)]), name="comb_stage_output")
    
        comb_memory = Memory(width=data_width, depth=number_of_pipelines)
        comb_write_memory_port = comb_memory.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        comb_read_memory_port = comb_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += comb_memory
        self.specials += comb_write_memory_port
        self.specials += comb_read_memory_port
                                              
        pipeline_counter = Signal(min=0, max=(2 * number_of_pipelines)) # need to double it because intermediate calculations otherwise would overflow

        write_offset = -2 % number_of_pipelines
        read_offset = -1 % number_of_pipelines

        write_counter = Signal(min=0, max=number_of_pipelines, reset=write_offset)
        read_counter = Signal(min=0, max=number_of_pipelines, reset=read_offset)

        self.sync += [
            comb_write_memory_port.we.eq(self.sink.valid),
            comb_write_memory_port.adr.eq(write_counter),

            comb_read_memory_port.re.eq(self.sink.valid),
            comb_read_memory_port.adr.eq(read_counter),
        ]

        self.sync += [
            If((self.sink.valid),
                If((write_counter == (number_of_pipelines - 1)),
                    write_counter.eq(0),
                ).Else(
                    write_counter.eq(write_counter + 1),
                ),
                If((read_counter == (number_of_pipelines - 1)),
                    read_counter.eq(0),
                ).Else(
                    read_counter.eq(read_counter + 1),
                ),
            ),
        ]

        valid_delay = Signal()

        self.sync += [
            valid_delay.eq(self.sink.valid),
            If((self.sink.valid),
                comb_write_memory_port.dat_w.eq(self.sink.data),
            ),
        ]
        
        self.sync += [
            self.source.data.eq(comb_write_memory_port.dat_w - comb_read_memory_port.dat_r),
            self.source.valid.eq(valid_delay),
        ]

class CICFilter(LiteXModule):
    def __init__(self, number_of_pipelines, number_of_inputs, input_data_width, number_of_stages, decimation_ratio, clock_domain):
        self.register_max_bit_width = input_data_width + math.ceil(number_of_stages * math.log2(decimation_ratio))

        self.sinks = [Endpoint(EndpointDescription([("data", (input_data_width + 1), True)])) for _ in range(number_of_inputs)]
        self.source = Endpoint(EndpointDescription([("data", self.register_max_bit_width, True)]))

        max_number_of_pipelines_per_input = ((number_of_pipelines + number_of_inputs - 1) // number_of_inputs)
        number_of_pipelines_left = number_of_pipelines
        self.integrator_chains = []
        self.decimators = []
        for _ in range(number_of_inputs):
            integrator_chain = []
            current_number_of_pipelines = min(number_of_pipelines_left, max_number_of_pipelines_per_input)
            number_of_pipelines_left -= current_number_of_pipelines
            for _ in range(number_of_stages):
                integrator_stage = CICFilterIntegratorStage(
                    number_of_pipelines=max(current_number_of_pipelines, 2),
                    data_width=self.register_max_bit_width,
                    clock_domain=clock_domain,
                )
                self.submodules += integrator_stage
                integrator_chain.append(integrator_stage)
            self.integrator_chains.append(integrator_chain)

            decimator = Decimator(
                number_of_pipelines=current_number_of_pipelines,
                decimation_ratio=decimation_ratio,
                data_width=self.register_max_bit_width,
            )
            self.submodules += decimator
            self.decimators.append(decimator)

        self.multiplexer = BurstMultiplexer(
            number_of_inputs=number_of_inputs,
            data_width=self.register_max_bit_width,
        )

        self.comb_chain = []
        for _ in range(number_of_stages):
            comb_stage = CICFilterCombStage(
                number_of_pipelines=number_of_pipelines,
                data_width=self.register_max_bit_width,
                clock_domain=clock_domain,
            )
            self.submodules += comb_stage
            self.comb_chain.append(comb_stage)

        for input_index in range(number_of_inputs):
            self.sync += [
                self.sinks[input_index].connect(self.integrator_chains[input_index][0].sink),
            ]

            for stage_index in range(number_of_stages - 1):
                self.sync += [
                    self.integrator_chains[input_index][stage_index].source.connect(self.integrator_chains[input_index][stage_index + 1].sink),
                ]

            self.sync += [
                self.integrator_chains[input_index][-1].source.connect(self.decimators[input_index].sink),
                self.decimators[input_index].source.connect(self.multiplexer.sinks[input_index]),
            ]

        self.sync += [
            self.multiplexer.source.connect(self.comb_chain[0].sink),
        ]

        for stage_index in range(number_of_stages - 1):
            self.sync += [
                self.comb_chain[stage_index].source.connect(self.comb_chain[stage_index + 1].sink),
            ]

        self.sync += [
            self.comb_chain[-1].source.connect(self.source),
        ]
