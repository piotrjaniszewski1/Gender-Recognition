#!/usr/bin/env python3

from wave import *
from struct import *
from pylab import *
import sys

chunk_duration = 20 #[s]

general_freq_band = 8000
section_freq_band= 50
sections_number = general_freq_band // section_freq_band

def prepare_samples(file_name):
    wav_reader = open(file_name, 'r')
    (_, sampwidth, framerate, nframes, _, _) = wav_reader.getparams()

    frame_data = wav_reader.readframes(nframes)
    samples_number = len(frame_data) // sampwidth

    format = {1: "%db", 2: "<%dh", 4: "<%dl"}[sampwidth] % samples_number

    return unpack(format, frame_data), framerate


def create_data_chunks(sampled_data, freq):
    chunk_length = int(floor(freq * chunk_duration))
    data_length = len(sampled_data)

    return [sampled_data[x:x + chunk_length] for x in range(0, data_length, chunk_length)]

def find_f0(chunked_data):
    max = 0
    argmax = 0

    for chunk in chunked_data:
        data = abs(np.fft.fft(chunk))
        data = log(data)
        data = np.fft.fft(data)
        if max < data[np.argmax(data)]:
            max = data[np.argmax(data)]
            argmax = np.argmax(data)

    return argmax

def create_energy_distribution(chunked_data):
    energy_distribution = list()

    for chunk in chunked_data:
        chunk_length = len(chunk)
        samples_in_band_number = chunk_length / sections_number
        local_summary = np.zeros(sections_number)

        data = abs(np.fft.fft(chunk))

        for i in range(chunk_length):
            local_summary[int(floor((i / samples_in_band_number)))] += data[i]

        energy_distribution.append(local_summary)

    return energy_distribution


def get_distribution(f):
    file_name = f

    sampled_data, framerate = prepare_samples(file_name)
    data_chunks = create_data_chunks(list(sampled_data), framerate)
    energy_distribution = create_energy_distribution(data_chunks)
    return energy_distribution[0]
