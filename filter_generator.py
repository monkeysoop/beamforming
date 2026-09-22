from filter_validator import validate_chain

import numpy as np
from scipy import signal



def calculate_cic_ripple_and_attenuation(responses, sample_rate, passband_edge, stopband_edge, decimation_ratio, number_of_frequencies):
    number_of_passband_samples = int(number_of_frequencies * (passband_edge / (sample_rate / 2.0))) + 1

    attenuation = 100000000.0
    for i in range(1, ((decimation_ratio // 2) + 1)):
        image_center_frequency = i * (sample_rate / decimation_ratio)
        image_start_frequency = max((image_center_frequency - stopband_edge), stopband_edge)
        image_stop_frequency = min((image_center_frequency + stopband_edge), (sample_rate / 2.0))
        image_start_index = int(number_of_frequencies * (image_start_frequency / (sample_rate / 2.0)))
        image_stop_index = int(number_of_frequencies * (image_stop_frequency / (sample_rate / 2.0)))
        if ((image_start_index < number_of_frequencies) and (image_start_index < image_stop_index)):
            image_attenuation = -20.0 * np.log10(np.max(np.maximum(np.abs(responses[image_start_index:image_stop_index]), 1.0e-12)))
            attenuation = min(attenuation, image_attenuation)

    ripple = 20.0 * np.log10(np.max(1.0 + np.abs(np.abs(responses[:number_of_passband_samples]) - 1.0)))

    return ripple, attenuation

def calculate_fir_ripple_and_attenuation(responses, sample_rate, passband_edge, stopband_edge):
    number_of_passband_samples = int(responses.shape[0] * (passband_edge / (sample_rate / 2.0))) + 1
    number_of_stopband_samples = int(responses.shape[0] * (((sample_rate / 2.0) - stopband_edge) / (sample_rate / 2.0)))
    stopband_start_index = responses.shape[0] - number_of_stopband_samples

    ripple = 20.0 * np.log10(np.max(1.0 + np.abs(np.abs(responses[:number_of_passband_samples]) - 1.0)))
    attenuation = -20.0 * np.log10(np.max(np.maximum(np.abs(responses[stopband_start_index:]), 1.0e-12)))

    return ripple, attenuation

def get_cic_response(w, number_of_cic_stages, decimation_ratio):
    gain = decimation_ratio**number_of_cic_stages
    responses = np.power(np.abs(np.sin(w / 2.0) / np.sin(w / (2.0 * decimation_ratio))), number_of_cic_stages) / gain

    return responses

def design_cic_filter(sample_rate, number_of_cic_stages, decimation_ratio, number_of_frequencies):
    frequencies = np.linspace(0.0, (sample_rate / 2.0), number_of_frequencies)

    w = np.linspace(0.0, (np.pi * decimation_ratio), number_of_frequencies) + 1.0e-12

    responses = get_cic_response(w, number_of_cic_stages, decimation_ratio)

    return frequencies, responses

def design_cic_compensation_filter(sample_rate, stopband_edge, bit_depth, number_of_taps, number_of_cic_stages, decimation_ratio, number_of_frequencies):
    frequencies, responses = design_cic_filter(sample_rate, number_of_cic_stages, 1, number_of_frequencies)
    responses = (1.0 / responses)

    number_of_stopband_samples = int(responses.shape[0] * (((sample_rate / 2.0) - stopband_edge) / (sample_rate / 2.0)))
    stopband_start_index = responses.shape[0] - number_of_stopband_samples
    responses[stopband_start_index:] = 0.0
    responses[-1] = 0.0

    taps = signal.firwin2(
        numtaps=number_of_taps,
        freq=frequencies,
        gain=responses,
        nfreqs=None,
        window="hamming",
        antisymmetric=False,
        fs=sample_rate,
    )

    quantization_multiplier = 2**(bit_depth - 1)
    taps = np.round(quantization_multiplier * taps)

    wideband_taps = np.zeros(number_of_taps * decimation_ratio)
    wideband_taps[::decimation_ratio] = taps

    frequencies, responses = signal.freqz(
        b=wideband_taps,
        a=1,
        worN=number_of_frequencies,
        whole=False,
        plot=None,
        fs=sample_rate,
    )

    responses /= quantization_multiplier

    return frequencies, responses

def design_fir_lowpass_filter(sample_rate, passband_edge, stopband_edge, bit_depth, number_of_taps, remez_iteration_limit, remez_grid_densitiy, frequencies_max, number_of_frequencies):
    try:
        taps = signal.remez(
            numtaps=number_of_taps,
            bands=[0.0, passband_edge, stopband_edge, (0.5 * sample_rate)],
            desired=[1.0, 0.0],
            weight=None,
            type="bandpass",
            maxiter=remez_iteration_limit,
            grid_density=remez_grid_densitiy,
            fs=sample_rate,
        )
    except ValueError:
        return None

    quantization_multiplier = 2**(bit_depth - 1)
    taps = np.round(quantization_multiplier * taps)

    frequencies, responses = signal.freqz(
        b=taps,
        a=1,
        worN=np.linspace(0.0, (sample_rate / 2.0), number_of_frequencies),
        whole=False,
        plot=None,
        fs=sample_rate,
        include_nyquist=True,
    )

    responses /= quantization_multiplier

    combined_frequencies, combined_responses = signal.freqz(
        b=taps,
        a=1,
        worN=np.linspace(0.0, frequencies_max, number_of_frequencies),
        whole=False,
        plot=None,
        fs=sample_rate,
        include_nyquist=True,
    )

    combined_responses /= quantization_multiplier

    ripple, attenuation = calculate_fir_ripple_and_attenuation(
        responses=responses,
        sample_rate=sample_rate,
        passband_edge=passband_edge,
        stopband_edge=stopband_edge,
    )

    return combined_frequencies, combined_responses, ripple, attenuation, taps

def design_compensation_filter(responses, sample_rate, stopband_edge, bit_depth, number_of_taps, frequencies_max, number_of_frequencies):
    number_of_frequencies_under_sample_rate = int(responses.shape[0] * ((sample_rate / 2.0) / frequencies_max))

    frequencies = np.linspace(0.0, (sample_rate / 2.0), number_of_frequencies_under_sample_rate)
    desired_responses = (1.0 / responses[:number_of_frequencies_under_sample_rate])

    number_of_stopband_samples = int(number_of_frequencies_under_sample_rate * (((sample_rate / 2.0) - stopband_edge) / (sample_rate / 2.0)))
    stopband_start_index = number_of_frequencies_under_sample_rate - number_of_stopband_samples
    desired_responses[stopband_start_index:] = 0.0
    desired_responses[-1] = 0.0

    taps = signal.firwin2(
        numtaps=number_of_taps,
        freq=frequencies,
        gain=desired_responses,
        nfreqs=None,
        window="hamming",
        antisymmetric=False,
        fs=sample_rate,
    )

    quantization_multiplier = 2**(bit_depth - 1)
    taps = np.round(quantization_multiplier * taps)

    frequencies, responses = signal.freqz(
        b=taps,
        a=1,
        worN=np.linspace(0.0, frequencies_max, number_of_frequencies),
        whole=False,
        plot=None,
        fs=sample_rate,
        include_nyquist=True,
    )

    responses /= quantization_multiplier

    return frequencies, responses, taps

def get_number_of_fir_taps(number_of_multipliers, system_clock_frequency, data_clock_frequency, number_of_microphones):
    return int(number_of_multipliers * ((system_clock_frequency // data_clock_frequency) // number_of_microphones))
def get_number_of_linear_fir_taps(number_of_multipliers, system_clock_frequency, data_clock_frequency, number_of_microphones):
    return (2 * get_number_of_fir_taps(number_of_multipliers, system_clock_frequency, data_clock_frequency, number_of_microphones) - 1)
def get_factor_sets(number, current=[], factor_sets=[]):
    if (number == 1):
        factor_sets.append(current)
        return factor_sets

    for i in range(2, (number + 1)):
        if ((number % i) == 0):
            factor_sets = get_factor_sets((number // i), (current + [i]), factor_sets)
    return factor_sets
def get_sum_sets(length, number, current=[], sum_sets=[], startFrom=2):
    #if (number == 0):
    if (len(current) == length):
        sum_sets.append(current)
        return sum_sets

    for i in range(startFrom, (number + 1)):
        if (len(current) == (length - 1)):
            return get_sum_sets(length, 0, (current + [number]), sum_sets)
        else:
            sum_sets = get_sum_sets(length, (number - i), (current + [i]), sum_sets)
    return sum_sets
def get_decimation_combinations(overall_decimatio_ratio):
    decimation_factor_sets = get_factor_sets(overall_decimatio_ratio, [], [])
    decimation_factor_sets_with_compensation = [i + [1] for i in decimation_factor_sets]
    return (decimation_factor_sets + decimation_factor_sets_with_compensation)
def get_multiplier_combinations(number_of_stages, number_of_multipliers):
    return get_sum_sets(number_of_stages, number_of_multipliers, [], [])

def design_chain(
    cic_decimation_ratio,
    number_of_cic_stages,
    sample_rate,
    passband_edge,
    stopband_edge,
    fir_decimation_ratios,
    number_of_multipliers_set,
    system_clock_frequency,
    number_of_microphones,
    bit_depth,
    remez_iteration_limit,
    remez_grid_densitiy,
    compensation_cutoff_factor,
    number_of_frequencies,
):
    firs_sample_frequency = sample_rate / cic_decimation_ratio
#    cic_frequencies, cic_responses = design_cic_filter(
    _, cic_responses = design_cic_filter(
        sample_rate=sample_rate,
        number_of_cic_stages=number_of_cic_stages,
        decimation_ratio=cic_decimation_ratio,
        number_of_frequencies=(number_of_frequencies + 1),
    )

#    cic_ripple, cic_attenuation = calculate_cic_ripple_and_attenuation(
    _, _ = calculate_cic_ripple_and_attenuation(
        responses=cic_responses,
        sample_rate=sample_rate,
        passband_edge=passband_edge,
        stopband_edge=stopband_edge,
        decimation_ratio=cic_decimation_ratio,
        number_of_frequencies=(number_of_frequencies + 1),
    )

    combined_frequencies = np.linspace(0, (sample_rate / 2.0), (number_of_frequencies + 1))
    combined_response = np.copy(cic_responses)

    current_fir_sample_frequency = (sample_rate / cic_decimation_ratio)

    firs_taps = []

    for fir_decimation_ratio, number_of_multipliers in zip(fir_decimation_ratios, number_of_multipliers_set):
        current_fir_output_sample_frequency = (current_fir_sample_frequency / fir_decimation_ratio)
        number_of_taps = get_number_of_fir_taps(number_of_multipliers, system_clock_frequency, current_fir_sample_frequency, number_of_microphones)

        fir_responses = None
        fir_taps = None
        if (fir_decimation_ratio == 1):
            _, fir_responses, fir_taps = design_compensation_filter(
                responses=combined_response,
                sample_rate=current_fir_output_sample_frequency,
                stopband_edge=(passband_edge + compensation_cutoff_factor * ((current_fir_output_sample_frequency / 2.0) - passband_edge)),
                bit_depth=bit_depth,
                number_of_taps=number_of_taps,
                frequencies_max=(sample_rate / 2.0),
                number_of_frequencies=(number_of_frequencies + 1)
            )
        else:
            fir_design_result = design_fir_lowpass_filter(
                sample_rate=current_fir_sample_frequency,
                passband_edge=passband_edge,
                stopband_edge=(current_fir_output_sample_frequency - stopband_edge),
                bit_depth=bit_depth,
                number_of_taps=number_of_taps,
                remez_iteration_limit=remez_iteration_limit,
                remez_grid_densitiy=remez_grid_densitiy,
                frequencies_max=(sample_rate / 2.0),
                number_of_frequencies=(number_of_frequencies + 1),
            )

            if (fir_design_result is None):
                return None

            _, fir_responses, _, _, fir_taps = fir_design_result

        combined_response *= np.abs(fir_responses)
        current_fir_sample_frequency = current_fir_output_sample_frequency

        firs_taps.append(fir_taps)

    combined_ripple, combined_attenuation = calculate_fir_ripple_and_attenuation(combined_response, sample_rate, passband_edge, stopband_edge)

    return combined_frequencies, combined_response, combined_ripple, combined_attenuation, firs_taps

def get_chain_parameters(pdm_sample_frequency, output_frequency, max_number_of_cic_stages, number_of_multipliers, compensation_filter_iterations_limit):
    overall_decimatio_ratio = int(pdm_sample_frequency / output_frequency)
    possible_cic_decimation_ratios = [i for i in range(2, (overall_decimatio_ratio + 1)) if ((overall_decimatio_ratio % i) == 0)]

    parameters = []

    for cic_decimation_ratio in possible_cic_decimation_ratios:
        overall_firs_decimation_ratio = int(overall_decimatio_ratio / cic_decimation_ratio)
        for number_of_cic_stages in range(1, (max_number_of_cic_stages + 1)):
            for fir_decimation_ratios in get_decimation_combinations(overall_firs_decimation_ratio):
                for number_of_multipliers_set in get_multiplier_combinations(len(fir_decimation_ratios), number_of_multipliers):
                    compensation_cutoff_factors = [None]
                    if (1 in fir_decimation_ratios):
                        compensation_cutoff_factors = np.linspace(0.0, 1.0, compensation_filter_iterations_limit)
                    for compensation_cutoff_factor in compensation_cutoff_factors:
                        parameters.append((cic_decimation_ratio, number_of_cic_stages, fir_decimation_ratios, number_of_multipliers_set, compensation_cutoff_factor))

    return parameters

SYSTEM_CLOCK_FREQUENCY = 60000000
OUTPUT_FREQUENCY = 48000
PDM_SAMPLE_FREQUENCY = 2400000.0
PASSBAND_EDGE_FREQUENCY = 10000.0
STOPBAND_EDGE_FREQUENCY = (OUTPUT_FREQUENCY / 2.0)

NUMBER_OF_MULTIPLIERS = 28
NUMBER_OF_MICROPHONES = 50
BIT_DEPTH = 18

MAX_NUMBER_OF_CIC_STAGES = 5
NUMBER_OF_FREQUENCIES = 10240
COMPENSATION_FILTER_ITERATIONS_LIMIT = 10
REMEZ_ITERATION_LIMIT = 100
REMEZ_GRID_DENSITY = 64

parameters = get_chain_parameters(
    pdm_sample_frequency=PDM_SAMPLE_FREQUENCY,
    output_frequency=OUTPUT_FREQUENCY,
    max_number_of_cic_stages=MAX_NUMBER_OF_CIC_STAGES,
    number_of_multipliers=NUMBER_OF_MULTIPLIERS,
    compensation_filter_iterations_limit=COMPENSATION_FILTER_ITERATIONS_LIMIT,
)

overall_number_of_iterations = len(parameters)
iteration = 0

results = []

for (cic_decimation_ratio, number_of_cic_stages, fir_decimation_ratios, number_of_multipliers_set, compensation_cutoff_factor) in parameters:
    fir_design_result = design_chain(
        cic_decimation_ratio=cic_decimation_ratio,
        number_of_cic_stages=number_of_cic_stages,
        sample_rate=PDM_SAMPLE_FREQUENCY,
        passband_edge=PASSBAND_EDGE_FREQUENCY,
        stopband_edge=STOPBAND_EDGE_FREQUENCY,
        fir_decimation_ratios=fir_decimation_ratios,
        number_of_multipliers_set=number_of_multipliers_set,
        system_clock_frequency=SYSTEM_CLOCK_FREQUENCY,
        number_of_microphones=NUMBER_OF_MICROPHONES,
        bit_depth=BIT_DEPTH,
        remez_iteration_limit=REMEZ_ITERATION_LIMIT,
        remez_grid_densitiy=REMEZ_GRID_DENSITY,
        compensation_cutoff_factor=compensation_cutoff_factor,
        number_of_frequencies=NUMBER_OF_FREQUENCIES,
    )

    iteration += 1

    if (fir_design_result is None):
        continue

    combined_frequencies, combined_response, combined_ripple, combined_attenuation, firs_taps = fir_design_result
    results.append([combined_ripple, combined_attenuation, (cic_decimation_ratio, number_of_cic_stages, fir_decimation_ratios, number_of_multipliers_set, compensation_cutoff_factor, firs_taps)])
    print("progress: ", round((100.0 * (iteration / overall_number_of_iterations)), 2), "%%    iteration:", iteration, "/", overall_number_of_iterations, "   ripple:", combined_ripple, "   attenuation:", combined_attenuation)
#    plt.plot(combined_frequencies, 20.0 * np.log10(np.maximum(np.abs(combined_response), 1.0e-12)))




results = sorted(results, key=(lambda x: x[1]), reverse=True)

for result in results[:20]:
    ripple, attenuation, (cic_decimation_ratio, number_of_cic_stages, fir_decimation_ratios, number_of_multipliers_set, compensation_cutoff_factor, firs_taps) = result
    print("ripple:", round(ripple, 4), "   attenuation:", round(attenuation, 2), "   cic ratio:", cic_decimation_ratio, "   cic stages:", number_of_cic_stages, "   fir ratios:", fir_decimation_ratios, "   multipliers:", number_of_multipliers_set, "   compensation cutoff:", compensation_cutoff_factor, "   firs taps:", [[int(i) for i in fir_taps] for fir_taps in firs_taps])

_, _, (cic_decimation_ratio, number_of_cic_stages, fir_decimation_ratios, _, _, firs_taps) = results[0]

synthetic_outputs, quantized_outputs, synthetic_gains, quantized_gains = validate_chain(
    sample_rate=PDM_SAMPLE_FREQUENCY,
    passband_edge=PASSBAND_EDGE_FREQUENCY,
    stopband_edge=STOPBAND_EDGE_FREQUENCY,
    bit_width=BIT_DEPTH,
    taps_bit_width=BIT_DEPTH,
    number_of_cic_stages=number_of_cic_stages,
    cic_decimation_ratio=cic_decimation_ratio,
    fir_decimation_ratios=fir_decimation_ratios,
    firs_taps=firs_taps,
    number_of_frequencies=NUMBER_OF_FREQUENCIES,
)

for synthetic_output, quantized_output, synthetic_gain, quantized_gain in zip(synthetic_outputs[-1:], quantized_outputs[-1:], synthetic_gains[-1:], quantized_gains[-1:]):
    normalized_synthetic_output = np.array(synthetic_output) / synthetic_gain
    synthetic_frequencies = np.fft.rfftfreq(normalized_synthetic_output.shape[0], d=(1.0 / OUTPUT_FREQUENCY))
    synthetic_response = np.fft.rfft(normalized_synthetic_output)

    normalized_quantized_output = np.array(quantized_output) / quantized_gain
    quantized_frequencies = np.fft.rfftfreq(normalized_quantized_output.shape[0], d=(1.0 / OUTPUT_FREQUENCY))
    quantized_response = np.fft.rfft(normalized_quantized_output)

    synthetic_ripple, _ = calculate_fir_ripple_and_attenuation(synthetic_response, OUTPUT_FREQUENCY, PASSBAND_EDGE_FREQUENCY, 0.0)
    quantized_ripple, _ = calculate_fir_ripple_and_attenuation(quantized_response, OUTPUT_FREQUENCY, PASSBAND_EDGE_FREQUENCY, 0.0)
    print("synthetic ripple:", round(synthetic_ripple, 4), "quantized ripple:", round(quantized_ripple, 4))
