from litex.gen import LiteXModule, Signal, If, Array, Memory, WRITE_FIRST

import math



class FIRFilterMultiplierBlock(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, number_of_taps, taps, taps_data_width, accumulator_data_width, clock_domain):
        self.input_data = Signal(data_width)
        self.input_data_valid = Signal()

        self.output_data = Signal(data_width)

        self.accumulated_data = Signal(accumulator_data_width)
        self.accumulated_data_valid = Signal()

        buffer = Memory(width=data_width, depth=(number_of_pipelines * number_of_taps))
        buffer_write_port = buffer.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        buffer_read_port = buffer.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += buffer
        self.specials += buffer_write_port
        self.specials += buffer_read_port

        taps_memory = Memory(width=taps_data_width, depth=number_of_taps, init=taps)
        taps_read_port = taps_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
        self.specials += taps_memory
        self.specials += taps_read_port

        self.taps_counter = Signal(min=0, max=(2 * number_of_taps), reset=((number_of_taps - 2) % number_of_taps))
        buffer_counter = Signal(min=0, max=(2 * (number_of_pipelines * number_of_taps)), reset=(((number_of_pipelines * number_of_taps) - 2) % (number_of_pipelines * number_of_taps)))

        buffer_write_offset = -1
        buffer_write_offset = buffer_write_offset % (number_of_pipelines * number_of_taps)

        buffer_read_offset = 0
        buffer_read_special_offset = number_of_taps

        taps_read_offset = 1

        self.comb += [
            buffer_write_port.we.eq(self.input_data_valid),

            If(((buffer_counter + buffer_write_offset) >= (number_of_pipelines * number_of_taps)),
                buffer_write_port.adr.eq(buffer_counter + (buffer_write_offset - (number_of_pipelines * number_of_taps))),
            ).Else(
                buffer_write_port.adr.eq(buffer_counter + buffer_write_offset),
            ),

            buffer_read_port.re.eq(self.input_data_valid),
            If((self.taps_counter == (number_of_taps - 1)),
                If(((buffer_counter + buffer_read_special_offset) >= (number_of_pipelines * number_of_taps)),
                    buffer_read_port.adr.eq(buffer_counter + (buffer_read_special_offset - (number_of_pipelines * number_of_taps))),
                ).Else(
                    buffer_read_port.adr.eq(buffer_counter + buffer_read_special_offset),
                ),
            ).Else(
                If(((buffer_counter + buffer_read_offset) >= (number_of_pipelines * number_of_taps)),
                    buffer_read_port.adr.eq(buffer_counter + (buffer_read_offset - (number_of_pipelines * number_of_taps))),
                ).Else(
                    buffer_read_port.adr.eq(buffer_counter + buffer_read_offset),
                ),
            ),

            taps_read_port.re.eq(self.input_data_valid),
            If(((self.taps_counter + taps_read_offset) >= number_of_taps),
                taps_read_port.adr.eq(self.taps_counter + (taps_read_offset - number_of_taps)),
            ).Else(
                taps_read_port.adr.eq(self.taps_counter + taps_read_offset),
            ),
        ]

        self.comb += [
            self.output_data.eq(buffer_read_port.dat_r),
        ]

        self.sync += [
            If((self.input_data_valid),
                If((self.taps_counter == 0),
                    buffer_write_port.dat_w.eq(self.input_data),
                    self.accumulated_data.eq(taps_read_port.dat_r * self.input_data),
                ).Else(
                    buffer_write_port.dat_w.eq(buffer_read_port.dat_r),
                    self.accumulated_data.eq(self.accumulated_data + (taps_read_port.dat_r * buffer_read_port.dat_r)),
                ),
                If((buffer_counter == ((number_of_pipelines * number_of_taps) - 1)),
                    buffer_counter.eq(0),
                ).Else(
                    buffer_counter.eq(buffer_counter + 1),
                ),
                If((self.taps_counter == (number_of_taps - 1)),
                    self.taps_counter.eq(0),
                    self.accumulated_data_valid.eq(1),
                ).Else(
                    self.taps_counter.eq(self.taps_counter + 1),
                    self.accumulated_data_valid.eq(0),
                ),
            ),
        ]

class FIRFilter(LiteXModule):
    def __init__(self, number_of_pipelines, data_width, number_of_multipliers, number_of_taps, taps, taps_data_width, accumulator_data_width, clock_domain):
        self.input_data = Signal(data_width)
        self.input_data_valid = Signal()
        self.input_data_ready = Signal()

        self.filtered_data = Signal(accumulator_data_width)
        self.filtered_data_valid = Signal()

        taps_per_multipliers = int(number_of_taps / number_of_multipliers)

        taps = [taps[i:(i + taps_per_multipliers)] for i in range(0, number_of_taps, taps_per_multipliers)]

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
            fir_blocks[0].input_data.eq(self.input_data),
            self.input_data_ready.eq(fir_blocks[0].taps_counter == (taps_per_multipliers - 1)),
        ]

        for i in range(1, len(fir_blocks)):
            self.comb += [
                fir_blocks[i].input_data.eq(fir_blocks[i - 1].output_data),
            ]

        for fir_block in fir_blocks:
            self.comb += [
                fir_block.input_data_valid.eq(self.input_data_valid),
            ]

        self.sync += [
            self.filtered_data_valid.eq(fir_blocks[0].accumulated_data_valid),
            If((fir_blocks[0].accumulated_data_valid),
                self.filtered_data.eq(sum([fir_block.accumulated_data for fir_block in fir_blocks])),
            ),
        ]
