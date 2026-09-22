import math



def decimator(datas, decimation_ratio):
    return datas[::decimation_ratio]

def quantizer(data, bit_width, target_bit_width):
    width_differene = (bit_width - target_bit_width)
    if (width_differene <= 0):
        return data

    quantized = None
    if (data >= 0):
        quantized = ((data + (1 << (width_differene - 1))) >> width_differene)
    else:
        quantized = -((-data + (1 << (width_differene - 1))) >> width_differene)

    return min(max(-int(2**(target_bit_width - 1)), quantized), (int(2**(target_bit_width - 1)) - 1))

def cic_filter(datas, number_of_cic_stages, decimation_ratio):
    integrators = [0 for _ in range(number_of_cic_stages)]
    integrators_delayed = [0 for _ in range(number_of_cic_stages)]
    combs = [[0 for _ in range(number_of_cic_stages)] for _ in range(decimation_ratio)]
    combs_delayed = [[0 for _ in range(number_of_cic_stages)] for _ in range(decimation_ratio)]

    output = []

    decimation_counter = 0
    for data in datas:
        integrators[0] = data + integrators_delayed[0]
        for i in range(1, number_of_cic_stages):
            integrators[i] = integrators[i - 1] + integrators_delayed[i]

        combs[decimation_counter][0] = integrators[number_of_cic_stages - 1]
        for i in range(1, number_of_cic_stages):
            combs[decimation_counter][i] = combs[decimation_counter][i - 1] - combs_delayed[decimation_counter][i - 1]

        output.append(combs[decimation_counter][number_of_cic_stages - 1] - combs_delayed[decimation_counter][number_of_cic_stages - 1])

        for i in range(number_of_cic_stages):
            combs_delayed[decimation_counter][i] = combs[decimation_counter][i]

        for i in range(number_of_cic_stages):
            integrators_delayed[i] = integrators[i]

        decimation_counter = (decimation_counter + 1) % decimation_ratio

    return output

def fir_filter(datas, taps):
    number_of_taps = len(taps)
    delays = [0 for _ in range(number_of_taps)]

    output = []
    for data in datas:
        for i in range((number_of_taps - 1), 0, -1):
            delays[i] = delays[i - 1]
        delays[0] = data

        accumulator = 0
        for i in range(number_of_taps):
            accumulator += taps[i] * delays[i]
        output.append(accumulator)
    return output

def validate_chain(
    sample_rate,
    passband_edge,
    stopband_edge,
    bit_width,
    taps_bit_width,
    number_of_cic_stages,
    cic_decimation_ratio,
    fir_decimation_ratios,
    firs_taps,
    number_of_frequencies,
):
    input_data = ([1] + [0 for _ in range(number_of_frequencies)])
    cic_filtered = cic_filter(input_data, number_of_cic_stages, cic_decimation_ratio)
    cic_decimated = decimator(cic_filtered, cic_decimation_ratio)

    cic_gain = ((cic_decimation_ratio**number_of_cic_stages) / cic_decimation_ratio)
    cic_bit_width = 1 + math.ceil(number_of_cic_stages * math.log2(cic_decimation_ratio))

    synthetic_outputs = [cic_decimated]
    quantized_outputs = [cic_decimated]
    synthetic_gains = [cic_gain]
    quantized_gains = [cic_gain]

    next_input_bit_width = cic_bit_width

    for fir_decimation_ratio, fir_taps in zip(fir_decimation_ratios, firs_taps):
        synthetic_input = synthetic_outputs[-1]
        quantized_input = [quantizer(i, next_input_bit_width, bit_width) for i in quantized_outputs[-1]]

        synthetic_output = decimator(fir_filter(synthetic_input, [int(i) for i in fir_taps]), fir_decimation_ratio)
        quantized_output = decimator(fir_filter(quantized_input, [int(i) for i in fir_taps]), fir_decimation_ratio)

        synthetic_fir_gain = (sum(fir_taps) / fir_decimation_ratio)
        quantized_fir_gain = ((sum(fir_taps) / fir_decimation_ratio) / int(2**max((next_input_bit_width - bit_width), 0)))

        next_input_bit_width = min(next_input_bit_width, bit_width) + math.ceil(math.log(sum(abs(fir_taps)), 2))

        synthetic_outputs.append(synthetic_output)
        quantized_outputs.append(quantized_output)
        synthetic_gains.append(synthetic_gains[-1] * synthetic_fir_gain)
        quantized_gains.append(quantized_gains[-1] * quantized_fir_gain)

    return synthetic_outputs, quantized_outputs, synthetic_gains, quantized_gains
