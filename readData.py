#!/usr/bin/env python3

from wave import *
from struct import *
from pylab import *
import sys

corePath = "train/"
chunkDuration = 20 #[s]

generalFreqBand = 8000
sectionFreqBand= 50
numberOfSections = generalFreqBand // sectionFreqBand

def prepareSamples(fileName):

    wavReader = open(fileName, 'r')
    (_, sampwidth, framerate, nframes, _, _) = wavReader.getparams()

    # in case the whole program works too slow, read every second frame (according to the article, it is enough)
    frameData = wavReader.readframes(nframes)
    samplesNumber = len(frameData) // sampwidth

    format = {1: "%db", 2: "<%dh", 4: "<%dl"}[sampwidth] % samplesNumber

    return unpack(format, frameData), framerate


def createDataChunks(sampledData, freq):
    chunkLength = int(floor(freq * chunkDuration))
    dataLength = len(sampledData)

    return [sampledData[x:x + chunkLength] for x in range(0, dataLength, chunkLength)]

def findF0(chunkedData):
    for chunk in chunkedData:
        data = abs(np.fft.fft(chunk))
        data = log(data)
        data = np.fft.fft(data)
        max = data[np.argmax(data)]

    return max

def createEnergyDistribution(chunkedData):

    energyDistribution = list() # the list will contain the energy distribution of each of the 20 milisecond parts of our wave file

    for chunk in chunkedData:
        chunkLength = len(chunk)
        numberOfSamplesInBand = chunkLength / numberOfSections
        localSummary = np.zeros(numberOfSections)

        data = abs(np.fft.fft(chunk))

        for i in range(chunkLength):
            # adding particular amplitudes has the same result as adding particular energies (the mass is unknown, so we can't count the energy)
            localSummary[int(floor((i / numberOfSamplesInBand)))] += data[i]
        energyDistribution.append(localSummary)
        break # FOR TESTING PURPOSES ONLY
    return energyDistribution


def getDistribution(f):

    fileName = f

    sampledData, framerate = prepareSamples(fileName)
    dataChunks = createDataChunks(list(sampledData), framerate)
    energyDistribution = createEnergyDistribution(dataChunks)
    return energyDistribution[0]
