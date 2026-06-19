from litex.gen import LiteXModule, Signal, If, Array

import math



class CICFilterIntegratorStagesSingle(LiteXModule):
    def __init__(self, number_of_cic_stages, decimation_ratio, register_min_value, register_max_value):
        self.integrators_input_data = Signal(min=register_min_value, max=register_max_value)
        self.integrators_output_data = Signal(min=register_min_value, max=register_max_value)

        self.integrators_input_data_valid = Signal()
        self.integrators_output_data_valid = Signal()

        integrators = Array(Signal(min=register_min_value, max=register_max_value) for _ in range(number_of_cic_stages))
        integrators_delayed = Array(Signal(min=register_min_value, max=register_max_value) for _ in range(number_of_cic_stages))

        counter = Signal(min=0, max=decimation_ratio)

        self.sync += [
            If((self.integrators_input_data_valid),
                integrators[0].eq(self.integrators_input_data + integrators_delayed[0]),
            ),
        ]

        for i in range(1, number_of_cic_stages):
            self.comb += [
                integrators[i].eq(integrators[i - 1] + integrators_delayed[i]),
            ]

        for i in range(number_of_cic_stages):
            self.sync += [
                If((self.integrators_input_data_valid),
                    integrators_delayed[i].eq(integrators[i]),
                ),
            ]

        self.sync += [
            If((self.integrators_input_data_valid & (counter == 0)),
                self.integrators_output_data.eq(integrators[number_of_cic_stages - 1]),
                self.integrators_output_data_valid.eq(1),
            ).Else(
                self.integrators_output_data_valid.eq(0),
            ),
        ]

        self.sync += [
            If((self.integrators_input_data_valid),
                If((counter == (decimation_ratio - 1)),
                    counter.eq(0),
                ).Else(
                    counter.eq(counter + 1),
                ),
            ),
        ]

class CICFilterCombStagesSingle(LiteXModule):
    def __init__(self, number_of_cic_stages, register_min_value, register_max_value):
        self.combs_input_data = Signal(min=register_min_value, max=register_max_value)
        self.combs_output_data = Signal(min=register_min_value, max=register_max_value)

        self.combs_input_data_valid = Signal()
        self.combs_output_data_valid = Signal()

        combs = Array(Signal(min=register_min_value, max=register_max_value) for _ in range(number_of_cic_stages))
        combs_delayed = Array(Signal(min=register_min_value, max=register_max_value) for _ in range(number_of_cic_stages))

        self.comb += [
            combs[0].eq(self.combs_input_data),
            self.combs_output_data.eq(combs[number_of_cic_stages - 1] - combs_delayed[number_of_cic_stages - 1]),
            self.combs_output_data_valid.eq(self.combs_input_data_valid),
        ]

        for i in range(1, number_of_cic_stages):
            self.comb += [
                combs[i].eq(combs[i - 1] - combs_delayed[i - 1]),
            ]

        for i in range(number_of_cic_stages):
            self.sync += [
                If((self.combs_input_data_valid),
                    combs_delayed[i].eq(combs[i]),
                ),
            ]

class CICFilterSingle(LiteXModule):
    def __init__(self, number_of_cic_stages, decimation_ratio, out_bit_depth):
        register_max_bit_width = 1 + math.ceil(number_of_cic_stages * math.log2(decimation_ratio))
        register_min_value = -1 * 2**(register_max_bit_width - 1)
        register_max_value = 2**(register_max_bit_width - 1) - 1

        self.pdm_data = Signal()
        self.pdm_data_valid = Signal()
        self.filtered_data = Signal(min=register_min_value, max=register_max_value)
        self.filtered_data_valid = Signal()

        self.submodules.integrator_stages = CICFilterIntegratorStagesSingle(
            number_of_cic_stages=number_of_cic_stages,
            decimation_ratio=decimation_ratio,
            register_min_value=register_min_value,
            register_max_value=register_max_value
        )

        self.submodules.comb_stages = CICFilterCombStagesSingle(
            number_of_cic_stages=number_of_cic_stages,
            register_min_value=register_min_value,
            register_max_value=register_max_value
        )

        self.comb += [
            self.integrator_stages.integrators_input_data.eq(self.pdm_data),
            self.integrator_stages.integrators_input_data_valid.eq(self.pdm_data_valid),

            self.comb_stages.combs_input_data.eq(self.integrator_stages.integrators_output_data),
            self.comb_stages.combs_input_data_valid.eq(self.integrator_stages.integrators_output_data_valid),

            self.filtered_data.eq(self.comb_stages.combs_output_data),
            self.filtered_data_valid.eq(self.comb_stages.combs_output_data_valid),
        ]
