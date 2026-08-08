from litex.gen import LiteXModule, Signal, If, Array, Memory, WRITE_FIRST

import math



class CICFilterIntegratorStages(LiteXModule):
    def __init__(self, number_of_pipelines, number_of_cic_stages, decimation_ratio, register_max_bit_width, register_min_value, register_max_value, clock_domain):
        self.integrators_input_data = Signal(min=register_min_value, max=register_max_value)
        self.integrators_input_data_valid = Signal()

        self.integrators_output_data = Signal(min=register_min_value, max=register_max_value)
        self.integrators_output_data_valid = Signal()

        integrator_write_memory_ports = []
        integrator_read_memory_ports = []

        for _ in range(number_of_cic_stages):
            integrator_memory = Memory(width=register_max_bit_width, depth=number_of_pipelines)
            integrator_write_memory_port = integrator_memory.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
            integrator_read_memory_port = integrator_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
            integrator_write_memory_ports.append(integrator_write_memory_port)
            integrator_read_memory_ports.append(integrator_read_memory_port)
            self.specials += integrator_memory
            self.specials += integrator_write_memory_port
            self.specials += integrator_read_memory_port
                                                 
        pipeline_counter = Signal(min=0, max=(2 * number_of_pipelines)) # need to double it because intermediate calculations otherwise would overflow
        decimation_counter = Signal(min=0, max=decimation_ratio)

        write_offset = -1 + 0
        read_offset = 1 + 0

        for cic_stage_index in range(number_of_cic_stages):
            cic_stage_write_offset = (write_offset - cic_stage_index) % number_of_pipelines
            cic_stage_read_offset = (read_offset - cic_stage_index) % number_of_pipelines
            self.comb += [
                integrator_write_memory_ports[cic_stage_index].we.eq(self.integrators_input_data_valid),
                If(((pipeline_counter + cic_stage_write_offset) >= number_of_pipelines),
                    integrator_write_memory_ports[cic_stage_index].adr.eq(pipeline_counter + (cic_stage_write_offset - number_of_pipelines)),
                ).Else(
                    integrator_write_memory_ports[cic_stage_index].adr.eq(pipeline_counter + cic_stage_write_offset),
                ),
                integrator_read_memory_ports[cic_stage_index].re.eq(self.integrators_input_data_valid),
                If(((pipeline_counter + cic_stage_read_offset) >= number_of_pipelines),
                    integrator_read_memory_ports[cic_stage_index].adr.eq(pipeline_counter + (cic_stage_read_offset - number_of_pipelines)),
                ).Else(
                    integrator_read_memory_ports[cic_stage_index].adr.eq(pipeline_counter + cic_stage_read_offset),
                ),
            ]

        self.integrators_output_index_offset = (write_offset - number_of_cic_stages + 1) % number_of_pipelines

        self.sync += [
            If((self.integrators_input_data_valid),
                integrator_write_memory_ports[0].dat_w.eq(integrator_read_memory_ports[0].dat_r + self.integrators_input_data),
            ),
        ]

        for cic_stage_index in range(1, number_of_cic_stages):
            self.sync += [
                If((self.integrators_input_data_valid),
                    integrator_write_memory_ports[cic_stage_index].dat_w.eq(integrator_read_memory_ports[cic_stage_index].dat_r + integrator_write_memory_ports[cic_stage_index - 1].dat_w),
                ),
            ]
        
        self.sync += [
            If((self.integrators_input_data_valid & (decimation_counter == 0)),
                self.integrators_output_data.eq(integrator_write_memory_ports[number_of_cic_stages - 1].dat_w),
                self.integrators_output_data_valid.eq(1),
            ).Else(
                self.integrators_output_data_valid.eq(0),
            ),
        ]

        self.sync += [
            If((self.integrators_input_data_valid),
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
            ),
        ]

class CICFilterCombStages(LiteXModule):
    def __init__(self, number_of_pipelines, number_of_cic_stages, register_max_bit_width, register_min_value, register_max_value, clock_domain):
        self.combs_input_data = Signal(min=register_min_value, max=register_max_value)
        self.combs_input_data_valid = Signal()

        self.combs_output_data = Signal(min=register_min_value, max=register_max_value)
        self.combs_output_data_valid = Signal()
        self.combs_output_data_pipeline_index = Signal(min=0, max=(2 * number_of_pipelines)) # need to double it because intermediate calculations otherwise would overflow

        comb_write_memory_ports = []
        comb_read_memory_ports = []

        for _ in range(number_of_cic_stages):
            comb_memory = Memory(width=register_max_bit_width, depth=number_of_pipelines)
            comb_write_memory_port = comb_memory.get_port(write_capable=True, async_read=False, has_re=False, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
            comb_read_memory_port = comb_memory.get_port(write_capable=False, async_read=False, has_re=True, we_granularity=0, mode=WRITE_FIRST, clock_domain=clock_domain)
            comb_write_memory_ports.append(comb_write_memory_port)
            comb_read_memory_ports.append(comb_read_memory_port)
            self.specials += comb_memory
            self.specials += comb_write_memory_port
            self.specials += comb_read_memory_port
                                              
        pipeline_counter = Signal(min=0, max=(2 * number_of_pipelines)) # need to double it because intermediate calculations otherwise would overflow

        write_offset = -2 + 0
        read_offset = -1 + 0

        for cic_stage_index in range(number_of_cic_stages):
            cic_stage_write_offset = (write_offset - cic_stage_index) % number_of_pipelines
            cic_stage_read_offset = (read_offset - cic_stage_index) % number_of_pipelines
            self.comb += [
                comb_write_memory_ports[cic_stage_index].we.eq(self.combs_input_data_valid),
                If(((pipeline_counter + cic_stage_write_offset) >= number_of_pipelines),
                    comb_write_memory_ports[cic_stage_index].adr.eq(pipeline_counter + (cic_stage_write_offset - number_of_pipelines)),
                ).Else(
                    comb_write_memory_ports[cic_stage_index].adr.eq(pipeline_counter + cic_stage_write_offset),
                ),
                comb_read_memory_ports[cic_stage_index].re.eq(self.combs_input_data_valid),
                If(((pipeline_counter + cic_stage_read_offset) >= number_of_pipelines),
                    comb_read_memory_ports[cic_stage_index].adr.eq(pipeline_counter + (cic_stage_read_offset - number_of_pipelines)),
                ).Else(
                    comb_read_memory_ports[cic_stage_index].adr.eq(pipeline_counter + cic_stage_read_offset),
                ),
            ]

        output_index_offset = (write_offset - number_of_cic_stages + 1) % number_of_pipelines

        self.sync += [
            If(((pipeline_counter + output_index_offset) >= number_of_pipelines),
                self.combs_output_data_pipeline_index.eq(pipeline_counter + (output_index_offset - number_of_pipelines)),
            ).Else(
                self.combs_output_data_pipeline_index.eq(pipeline_counter + output_index_offset),
            ),
        ]

        self.sync += [
            If((self.combs_input_data_valid),
                comb_write_memory_ports[0].dat_w.eq(self.combs_input_data),
            ),
        ]

        for cic_stage_index in range(1, number_of_cic_stages):
            self.sync += [
                If((self.combs_input_data_valid),
                    comb_write_memory_ports[cic_stage_index].dat_w.eq(comb_write_memory_ports[cic_stage_index - 1].dat_w - comb_read_memory_ports[cic_stage_index - 1].dat_r),
                ),
            ]
        
        self.sync += [
            If((self.combs_input_data_valid),
                self.combs_output_data.eq(comb_write_memory_ports[number_of_cic_stages - 1].dat_w - comb_read_memory_ports[number_of_cic_stages - 1].dat_r),
                self.combs_output_data_valid.eq(1),
            ).Else(
                self.combs_output_data_valid.eq(0),
            ),
        ]

        self.sync += [
            If((self.combs_input_data_valid),
                If((pipeline_counter == (number_of_pipelines - 1)),
                    pipeline_counter.eq(0),
                ).Else(
                    pipeline_counter.eq(pipeline_counter + 1),
                ),
            ),
        ]

class CICFilter(LiteXModule):
    def __init__(self, number_of_pipelines, number_of_cic_stages, decimation_ratio, clock_domain):
        register_max_bit_width = 2 + math.ceil(number_of_cic_stages * math.log2(decimation_ratio))
        register_min_value = -1 * 2**(register_max_bit_width - 1)
        register_max_value = 2**(register_max_bit_width - 1) - 1

        self.pdm_data = Signal(2)
        self.pdm_data_valid = Signal()
        self.filtered_data = Signal(min=register_min_value, max=register_max_value)
        self.filtered_data_valid = Signal()
        self.filtered_data_pipeline_index = Signal(min=0, max=number_of_pipelines)

        self.submodules.integrator_stages = CICFilterIntegratorStages(
            number_of_pipelines=number_of_pipelines,
            number_of_cic_stages=number_of_cic_stages,
            decimation_ratio=decimation_ratio,
            register_max_bit_width=register_max_bit_width,
            register_min_value=register_min_value,
            register_max_value=register_max_value,
            clock_domain=clock_domain,
        )

        self.submodules.comb_stages = CICFilterCombStages(
            number_of_pipelines=number_of_pipelines,
            number_of_cic_stages=number_of_cic_stages,
            register_max_bit_width=register_max_bit_width,
            register_min_value=register_min_value,
            register_max_value=register_max_value,
            clock_domain=clock_domain,
        )

        integrator_offset = (self.integrator_stages.integrators_output_index_offset + 1) % number_of_pipelines # +1 is because we use this inside a sync block which adds 1 more clock cycle of latency

        self.sync += [
            self.integrator_stages.integrators_input_data.eq(2 * self.pdm_data - 1),
            self.integrator_stages.integrators_input_data_valid.eq(self.pdm_data_valid),

            self.comb_stages.combs_input_data.eq(self.integrator_stages.integrators_output_data),
            self.comb_stages.combs_input_data_valid.eq(self.integrator_stages.integrators_output_data_valid),

            self.filtered_data.eq(self.comb_stages.combs_output_data),
            self.filtered_data_valid.eq(self.comb_stages.combs_output_data_valid),

            If(((self.comb_stages.combs_output_data_pipeline_index + integrator_offset) >= number_of_pipelines),
                self.filtered_data_pipeline_index.eq(self.comb_stages.combs_output_data_pipeline_index + (integrator_offset - number_of_pipelines)),
            ).Else(
                self.filtered_data_pipeline_index.eq(self.comb_stages.combs_output_data_pipeline_index + integrator_offset),
            ),
        ]
