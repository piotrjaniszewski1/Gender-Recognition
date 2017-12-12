#!/usr/bin/env python3

from wave import *
from struct import *
from pylab import *
import sys

corePath = "/home/piotr/Desktop/PycharmProjects/Gender-Recognition/recordings/"

freq = 44100
chunkDuration = 0.02 #[s]

generalFreqBand = 8000
sectionFreqBand= 50
numberOfSections = generalFreqBand // sectionFreqBand + 1

def prepareSamples(fileName):

    wavReader = open(fileName, 'r')
    (_, sampwidth, _, nframes, _, _) = wavReader.getparams()

    # in case the whole program works too slow, read every second frame (according to the article, it is enough)
    frameData = wavReader.readframes(nframes)
    samplesNumber = len(frameData) // sampwidth

    format = {1: "%db", 2: "<%dh", 4: "<%dl"}[sampwidth] % samplesNumber

    return unpack(format, frameData)


def createDataChunks(sampledData):
    chunkLength = int(freq * chunkDuration)
    dataLength = len(sampledData)

    return [sampledData[x:x + chunkLength] for x in range(0, dataLength, chunkLength)]


def createEnergyDistribution(chunkedData):

    chunkLenght = len(chunkedData[0])
    numberOfSamplesInBand = chunkLenght / numberOfSections

    energyDistribution = list() # the list will contain the energy distribution of each of the 20 milisecond parts of our wave file

    for x in chunkedData:
        data = abs(np.fft.fft(x))

        localSummary = np.zeros(numberOfSections)

        for i in range(chunkLenght):
            # adding particular amplitudes has the same result as adding particular energies (the mass is unknown, so we can't count the energy)
            localSummary[int(floor(i / numberOfSamplesInBand))] += data[i]

        energyDistribution.append(localSummary)

    return energyDistribution


def main():

    fileName = corePath + sys.argv[1]

    sampledData = prepareSamples(fileName)
    dataChunks = createDataChunks(list(sampledData))
    energyDistribution = createEnergyDistribution(dataChunks)

if __name__ == '__main__':
    main()