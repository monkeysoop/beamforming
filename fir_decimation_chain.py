from quantizer import Quantizer
from fir_filter import FIRFilter
from decimator import Decimator

from filter_validator import validate_chain

from litex.gen import LiteXModule, Signal, If
from litex.soc.interconnect.stream import Endpoint, EndpointDescription, SyncFIFO

from migen.sim import run_simulation

import math



class FirDecimationChain(LiteXModule):
    def __init__(self, number_of_pipelines, input_data_width, data_width, output_data_width, taps_data_width, fir_decimation_ratios, number_of_multipliers_set, firs_taps, clock_domain, quantizer_delay_count):
        number_of_stages = len(firs_taps)

        if (len(fir_decimation_ratios) != number_of_stages):
            return

        if (len(number_of_multipliers_set) != number_of_stages):
            return

        self.sink = Endpoint(EndpointDescription([("data", input_data_width)]), name="chain_input")
        self.source = Endpoint(EndpointDescription([("data", output_data_width)]), name="chain_output")

        self.fifo_data_widths = []
        self.fir_filter_input_widths = []
        self.fir_filter_output_widths = []
        self.quantizer_input_widths = [input_data_width]
        self.quantizer_output_widths = [data_width]
        self.decimator_data_widths = []

        intermediate_input_bit_width = input_data_width
        for i in range(number_of_stages):
            accumulator_data_width = min(intermediate_input_bit_width, data_width) + math.ceil(math.log(sum([abs(j) for j in firs_taps[i]]), 2))
            intermediate_output_bit_width = min(accumulator_data_width, data_width)

            self.fifo_data_widths.append(intermediate_input_bit_width)
            self.fir_filter_input_widths.append(intermediate_input_bit_width)
            self.fir_filter_output_widths.append(accumulator_data_width)
            self.quantizer_input_widths.append(accumulator_data_width)
            self.quantizer_output_widths.append(intermediate_output_bit_width)
            self.decimator_data_widths.append(intermediate_output_bit_width)

            intermediate_input_bit_width = intermediate_output_bit_width

        self.quantizer_output_widths[-1] = output_data_width
        self.decimator_data_widths[-1] = output_data_width

        self.fifos = []
        for fifo_data_width in self.fifo_data_widths:
            fifo = SyncFIFO(
                layout=[("data", fifo_data_width)],
                depth=number_of_pipelines,
                buffered=True
            )
            self.submodules += fifo
            self.fifos.append(fifo)

        self.fir_filters = []
        for fir_filter_input_width, fir_filter_output_width, number_of_multipliers, taps in zip(self.fir_filter_input_widths, self.fir_filter_output_widths, number_of_multipliers_set, firs_taps):
            fir_filter = FIRFilter(
                number_of_pipelines=number_of_pipelines,
                data_width=fir_filter_input_width,
                number_of_multipliers=number_of_multipliers,
                number_of_taps=len(taps),
                taps=taps,
                taps_data_width=taps_data_width,
                accumulator_data_width=fir_filter_output_width,
                clock_domain=clock_domain,
            )
            self.submodules += fir_filter
            self.fir_filters.append(fir_filter)

        self.quantizers = []
        for quantizer_input_width, quantizer_output_width in zip(self.quantizer_input_widths, self.quantizer_output_widths):
            quantizer = Quantizer(
                input_data_width=quantizer_input_width,
                output_data_width=quantizer_output_width,
                delay_count=quantizer_delay_count,
            )
            self.submodules += quantizer
            self.quantizers.append(quantizer)

        self.decimators = []
        for decimator_data_width, fir_decimation_ratio in zip(self.decimator_data_widths, fir_decimation_ratios):
            decimator = Decimator(
                number_of_pipelines=number_of_pipelines,
                decimation_ratio=fir_decimation_ratio,
                data_width=decimator_data_width,
            )
            self.submodules += decimator
            self.decimators.append(decimator)

        self.sync += [
            self.sink.connect(self.quantizers[0].sink),
            self.quantizers[0].source.connect(self.fifos[0].sink),
        ]

        for i in range(number_of_stages):
            self.sync += [
                self.fifos[i].source.ready.eq(self.fifos[i].source.valid & self.fir_filters[i].next_input_data_ready),
                self.fir_filters[i].sink.data.eq(self.fifos[i].source.data),
                self.fir_filters[i].sink.valid.eq(self.fifos[i].source.valid & self.fir_filters[i].next_input_data_ready),
                self.fir_filters[i].source.connect(self.quantizers[i + 1].sink),
                self.quantizers[i + 1].source.connect(self.decimators[i].sink),
            ]

        for i in range(number_of_stages - 1):
            self.sync += [
                self.decimators[i].source.connect(self.fifos[i + 1].sink),
            ]

        self.sync += [
            self.decimators[-1].source.connect(self.source),
        ]
