def decimator(datas, decimation_ratio):
    return datas[::decimation_ratio]

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

####        for i in range((number_of_cic_stages - 1), 0, -1):
####            integrators[i] = integrators[i - 1] + integrators_delayed[i]
####        integrators[0] = d + integrators_delayed[0]


#        print(hex(integrators[number_of_cic_stages - 1])[2:].upper())
        combs[decimation_counter][0] = integrators[number_of_cic_stages - 1]
        for i in range(1, number_of_cic_stages):
            combs[decimation_counter][i] = combs[decimation_counter][i - 1] - combs_delayed[decimation_counter][i - 1]

        output.append(combs[decimation_counter][number_of_cic_stages - 1] - combs_delayed[decimation_counter][number_of_cic_stages - 1])

####            output.append(combs[decimation_counter][number_of_cic_stages - 1] - combs_delayed[decimation_counter][number_of_cic_stages - 1])
####            for i in range((number_of_cic_stages - 1), 0, -1):
####                combs[decimation_counter][i] = combs[decimation_counter][i - 1] - combs_delayed[decimation_counter][i - 1]
####            combs[decimation_counter][0] = integrators[number_of_cic_stages - 1]

#        print("output:       ", output)
#        print("combs_delayed:", [hex(i)[2:].upper() for i in combs_delayed[decimation_counter]])
#        print("combs:        ", [hex(i)[2:].upper() for i in combs[decimation_counter]], hex(output[-1])[2:].upper())
#        print()

        for i in range(number_of_cic_stages):
            combs_delayed[decimation_counter][i] = combs[decimation_counter][i]

        for i in range(number_of_cic_stages):
            integrators_delayed[i] = integrators[i]

        decimation_counter = (decimation_counter + 1) % decimation_ratio

#        print("integrators:        ", [hex(i)[2:].upper() for i in integrators])
#        print("integrators_delayed:", [hex(i)[2:].upper() for i in integrators_delayed])
#        print()

    return output

def fir_filter(datas, taps):
    number_of_taps = len(taps)
    delays = [0 for _ in range(number_of_taps)]

    output = []
    for data in datas:
        for i in range((number_of_taps - 1), 0, -1):
            delays[i] = delays[i - 1]
        delays[0] = data

        accumulator = 0.0
        for i in range(number_of_taps):
            accumulator += taps[i] * delays[i]
        output.append(accumulator)
    return output

def validate_chain(
    sample_rate,
    passband_edge,
    stopband_edge,
    bit_depth,
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

    synthetic_outputs = [cic_decimated]
    quantized_outputs = [[(i / cic_gain) for i in cic_decimated]]
    gains = [cic_gain]

    for fir_decimation_ratio, fir_taps in zip(fir_decimation_ratios, firs_taps):
        quantization_multiplier = 2**(bit_depth - 1)

        synthetic_input = synthetic_outputs[-1]
        quantized_input = [(round(quantization_multiplier * i) / quantization_multiplier) for i in quantized_outputs[-1]]

        synthetic_output = decimator(fir_filter(synthetic_input, [int(i) for i in fir_taps]), fir_decimation_ratio)
        quantized_output = decimator(fir_filter(quantized_input, [int(i) for i in fir_taps]), fir_decimation_ratio)

        fir_gain = (sum(fir_taps) / fir_decimation_ratio)
        quantized_output = [(i / fir_gain) for i in quantized_output]

        synthetic_outputs.append(synthetic_output)
        quantized_outputs.append(quantized_output)
        gains.append(gains[-1] * fir_gain)

    return synthetic_outputs, quantized_outputs, gains