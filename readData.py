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

    frameData = wavReader.readframes(nframes)
    samplesNumber = len(frameData) // sampwidth

    format = {1: "%db", 2: "<%dh", 4: "<%dl"}[sampwidth] % samplesNumber

    return unpack(format, frameData), framerate


def createDataChunks(sampledData, freq):
    chunkLength = int(floor(freq * chunkDuration))
    dataLength = len(sampledData)

    return [sampledData[x:x + chunkLength] for x in range(0, dataLength, chunkLength)]

def findF0(chunkedData):
    max = 0
    argmax = 0

    for chunk in chunkedData:
        data = abs(np.fft.fft(chunk))
        data = log(data)
        data = np.fft.fft(data)
        if max < data[np.argmax(data)]:
            max = data[np.argmax(data)]
            argmax = np.argmax(data)

    return argmax

def createEnergyDistribution(chunkedData):

    energyDistribution = list()

    for chunk in chunkedData:
        chunkLength = len(chunk)
        numberOfSamplesInBand = chunkLength / numberOfSections
        localSummary = np.zeros(numberOfSections)

        data = abs(np.fft.fft(chunk))

        for i in range(chunkLength):
            localSummary[int(floor((i / numberOfSamplesInBand)))] += data[i]

        energyDistribution.append(localSummary)

    return energyDistribution


def getDistribution(f):

    fileName = f

    sampledData, framerate = prepareSamples(fileName)
    dataChunks = createDataChunks(list(sampledData), framerate)
    energyDistribution = createEnergyDistribution(dataChunks)
    return energyDistribution[0]
