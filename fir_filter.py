from litex.gen import LiteXModule, Signal, If, Array, Memory, WRITE_FIRST
from litex.soc.interconnect.stream import Endpoint, EndpointDescription

import math



class FIRFilterMultiplierBlock(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, number_of_taps, taps, taps_data_width, accumulator_data_width, clock_domain):
        self.sink = Endpoint(EndpointDescription([("data", data_width, True)]))
        self.source = Endpoint(EndpointDescription([("data", data_width, True)]))
        self.accumulator = Endpoint(EndpointDescription([("data", accumulator_data_width, True)]))

        self.next_input_data_ready = Signal()

        buffer = Memory(width=(data_width + 1), depth=((number_of_pipelines * number_of_taps) + 1)) # the +1 to the width is because it indicates if that data is valid or not, the +1 to depth is in case depth is 1 which would throw an error
        buffer_write_port = buffer.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        buffer_read_port = buffer.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += buffer
        self.specials += buffer_write_port
        self.specials += buffer_read_port

        taps_memory = Memory(width=taps_data_width, depth=(number_of_taps + 1), init=taps) # the +1 to depth is in case depth is 1 which would throw an error
        taps_read_port = taps_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += taps_memory
        self.specials += taps_read_port

        taps_counter = Signal(min=0, max=number_of_taps)

        buffer_write_offset = 0 % (number_of_pipelines * number_of_taps)
        buffer_read_offset = 1 % (number_of_pipelines * number_of_taps)
        buffer_read_special_offset = (number_of_taps + 1) % (number_of_pipelines * number_of_taps)
        taps_read_offset = 0 % number_of_taps

        buffer_write_counter = Signal(min=0, max=(number_of_pipelines * number_of_taps), reset=buffer_write_offset)
        buffer_read_counter = Signal(min=0, max=(number_of_pipelines * number_of_taps), reset=buffer_read_offset)
        buffer_read_special_counter = Signal(min=0, max=(number_of_pipelines * number_of_taps), reset=buffer_read_special_offset)
        taps_read_counter = Signal(min=0, max=number_of_taps, reset=taps_read_offset)

        run_pipeline = Signal()

        self.comb += [
            run_pipeline.eq((taps_counter != 0) | self.sink.valid),
        ]

        self.sync += [
            buffer_write_port.we.eq(run_pipeline),
            buffer_write_port.adr.eq(buffer_write_counter),

            buffer_read_port.re.eq(run_pipeline),
            If((taps_counter == (number_of_taps - 2)),
                buffer_read_port.adr.eq(buffer_read_special_counter),
            ).Else(
                buffer_read_port.adr.eq(buffer_read_counter),
            ),

            taps_read_port.re.eq(run_pipeline),
            taps_read_port.adr.eq(taps_read_counter),
        ]

        self.comb += [
            self.next_input_data_ready.eq((taps_counter == (number_of_taps - 1)) | (~run_pipeline)),
            self.source.data.eq(buffer_read_port.dat_r[:data_width]),
            self.source.valid.eq((taps_counter == 0) & buffer_read_port.dat_r[data_width]),
        ]

        data_temp_valid = Signal()

        tap_and_data_delay = 2
        tap_and_data_valid_delay_buffer = [Signal() for _ in range(tap_and_data_delay + 1)]
        data_delay_buffer = [Signal(bits_sign=(data_width, True)) for _ in range(tap_and_data_delay + 1)]
        tap_delay_buffer = [Signal(bits_sign=(taps_data_width, True)) for _ in range(tap_and_data_delay + 1)]

        multiplied_delay = 2
        multiplied_data_valid_delay_buffer = [Signal() for _ in range(multiplied_delay + 1)]
        multiplied_data_delay_buffer = [Signal(bits_sign=((taps_data_width + data_width), True)) for _ in range(multiplied_delay + 1)]

        multiplied_data_valid = Signal()
        multiplied_data = Signal(bits_sign=((taps_data_width + data_width), True))

        self.accumulator_counter = Signal(min=0, max=(number_of_taps + 1))

        self.sync += [
            If((taps_counter == 0),
                buffer_write_port.dat_w[:data_width].eq(self.sink.data),
                buffer_write_port.dat_w[data_width].eq(self.sink.valid),
                data_temp_valid.eq(self.sink.valid),
            ).Else(
                buffer_write_port.dat_w.eq(buffer_read_port.dat_r),
            ),
        ]

        self.comb += [
            tap_and_data_valid_delay_buffer[0].eq(data_temp_valid),
            data_delay_buffer[0].eq(buffer_write_port.dat_w[:data_width]),
            tap_delay_buffer[0].eq(taps_read_port.dat_r),
        ]

        for i in range(1, (tap_and_data_delay + 1)):
            self.sync += [
                tap_and_data_valid_delay_buffer[i].eq(tap_and_data_valid_delay_buffer[i - 1]),
                data_delay_buffer[i].eq(data_delay_buffer[i - 1]),
                tap_delay_buffer[i].eq(tap_delay_buffer[i - 1]),
            ]

        self.sync += [
            multiplied_data_valid_delay_buffer[0].eq(tap_and_data_valid_delay_buffer[-1]),
            multiplied_data_delay_buffer[0].eq(data_delay_buffer[-1] * tap_delay_buffer[-1]),
        ]

        for i in range(1, (multiplied_delay + 1)):
            self.sync += [
                multiplied_data_valid_delay_buffer[i].eq(multiplied_data_valid_delay_buffer[i - 1]),
                multiplied_data_delay_buffer[i].eq(multiplied_data_delay_buffer[i - 1]),
            ]

        self.comb += [
            multiplied_data_valid.eq(multiplied_data_valid_delay_buffer[-1]),
            multiplied_data.eq(multiplied_data_delay_buffer[-1]),
        ]

        self.sync += [
            If((multiplied_data_valid),
                If((self.accumulator_counter == 0),
                    self.accumulator.data.eq(multiplied_data),
                ).Else(
                    self.accumulator.data.eq(self.accumulator.data + multiplied_data),
                ),
                If((self.accumulator_counter == (number_of_taps - 1)),
                    self.accumulator_counter.eq(0),
                    self.accumulator.valid.eq(1),
                ).Else(
                    self.accumulator_counter.eq(self.accumulator_counter + 1),
                    self.accumulator.valid.eq(0),
                ),
            ).Else(
                self.accumulator.valid.eq(0),
            ),

            If((run_pipeline),
                If((buffer_write_counter == ((number_of_pipelines * number_of_taps) - 1)),
                    buffer_write_counter.eq(0),
                ).Else(
                    buffer_write_counter.eq(buffer_write_counter + 1),
                ),
                If((buffer_read_counter == ((number_of_pipelines * number_of_taps) - 1)),
                    buffer_read_counter.eq(0),
                ).Else(
                    buffer_read_counter.eq(buffer_read_counter + 1),
                ),
                If((buffer_read_special_counter == ((number_of_pipelines * number_of_taps) - 1)),
                    buffer_read_special_counter.eq(0),
                ).Else(
                    buffer_read_special_counter.eq(buffer_read_special_counter + 1),
                ),
                If((taps_read_counter == (number_of_taps - 1)),
                    taps_read_counter.eq(0),
                ).Else(
                    taps_read_counter.eq(taps_read_counter + 1),
                ),
                If((taps_counter == (number_of_taps - 1)),
                    taps_counter.eq(0),
                ).Else(
                    taps_counter.eq(taps_counter + 1),
                ),
            ),
        ]

class FIRFilter(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, number_of_multipliers, number_of_taps, taps, taps_data_width, accumulator_data_width, clock_domain):
        self.sink = Endpoint(EndpointDescription([("data", data_width, True)]))
        self.source = Endpoint(EndpointDescription([("data", accumulator_data_width, True)]))

        self.next_input_data_ready = Signal()

        taps_per_multipliers = int(number_of_taps / number_of_multipliers)

        taps = [taps[i:(i + taps_per_multipliers)] for i in range(0, number_of_taps, taps_per_multipliers)]
        taps = [(tap_block[1:] + [tap_block[0]]) for tap_block in taps]

        fir_blocks = []
        for i in range(number_of_multipliers):
            fir_block = FIRFilterMultiplierBlock(
               number_of_pipelines=number_of_pipelines,
               data_width=data_width,
               number_of_taps=taps_per_multipliers,
               taps=taps[i],
               taps_data_width=taps_data_width,
               accumulator_data_width=accumulator_data_width,
               clock_domain=clock_domain, 
            )
            self.submodules += fir_block
            fir_blocks.append(fir_block)

        self.comb += [
            fir_blocks[0].sink.data.eq(self.sink.data),
            fir_blocks[0].sink.valid.eq(self.sink.valid),
            self.next_input_data_ready.eq(fir_blocks[0].next_input_data_ready),
        ]

        self.sync += [
            self.sink.ready.eq(self.next_input_data_ready),
        ]

        for i in range(1, number_of_multipliers):
            self.comb += [
                fir_blocks[i].sink.data.eq(fir_blocks[i - 1].source.data),
                fir_blocks[i].sink.valid.eq(fir_blocks[i - 1].source.valid),
            ]

        accumulator_tree_split_factor = 2
        number_of_accumulator_tree_levels = (math.ceil(math.log(number_of_multipliers, accumulator_tree_split_factor)) + 1)

        accumulator_tree = [[Signal(accumulator_data_width) for _ in range(math.ceil(number_of_multipliers / (accumulator_tree_split_factor**i)))] for i in range(number_of_accumulator_tree_levels)]
        valid_chain = [Signal() for _ in range(number_of_accumulator_tree_levels)]

        for level_index in range(1, number_of_accumulator_tree_levels):
            self.sync += [
                valid_chain[level_index].eq(valid_chain[level_index - 1]),
            ]
            current_level = accumulator_tree[level_index]
            level_below = accumulator_tree[level_index - 1]
            for accumulator_index in range(len(current_level)):
                accumulators_below = [level_below[accumulator_tree_split_factor * accumulator_index + i] for i in range(accumulator_tree_split_factor) if (((accumulator_tree_split_factor * accumulator_index) + i) < len(level_below))]
                self.sync += [
                    current_level[accumulator_index].eq(sum(accumulators_below)),
                ]

        self.comb += [
            valid_chain[0].eq(fir_blocks[0].accumulator.valid),
        ]

        for i in range(number_of_multipliers):
            self.comb += [
                accumulator_tree[0][i].eq(fir_blocks[i].accumulator.data),
            ]

        self.comb += [
            self.source.valid.eq(valid_chain[-1]),
            self.source.data.eq(accumulator_tree[-1][0]),
        ]
