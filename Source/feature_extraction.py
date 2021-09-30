import soundfile as sf
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard
import json
import os
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt




def left_channel(audio):

    audio = np.array(audio.T[0], dtype=np.float32).T

    return audio


def block_audio(x,blockSize,hopSize,fs):
    """
    for blocking the input audio signal into overlapping blocks

    returns:
        a matrix xb (dimension NumOfBlocks X blockSize)
        a vector timeInSec (dimension NumOfBlocks)

    timeInSec will refer to the start time of each block.
    """
    timeInSample = np.array([])
    currentTimeInSample = 0
    xb = None

    while currentTimeInSample < x.size:
        if currentTimeInSample+blockSize > x.size:

            newBlock = np.append(x[currentTimeInSample:], np.zeros(blockSize-(x.size-currentTimeInSample)))[np.newaxis, :]
        else:
            newBlock = x[currentTimeInSample : currentTimeInSample+blockSize][np.newaxis, :]

        if xb is not None:
            xb = np.concatenate((xb, newBlock), axis=0)
            print(xb.shape)
        else:
            xb = newBlock

        timeInSample = np.append(timeInSample, currentTimeInSample)
        currentTimeInSample += hopSize

    timeInSec = timeInSample / fs
    return xb, timeInSec




def feature_extraction(audio_path, filename):

    audio, sampleRate = sf.read(audio_path + "/" + filename)

    audio = left_channel(audio)

    w = essentia.standard.Windowing(type = 'hann')
    spectrum = essentia.standard.Spectrum()
    mfcc = essentia.standard.MFCC()
    logNorm = essentia.standard.UnaryOperator(type='log')

    mfccs = []
    melbands = []
    melbands_log = []


    for frame in block_audio(audio, 1024, 512, sampleRate)[0]:
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(logNorm(mfcc_bands))

        # transpose to have it in a better shape
        # we need to convert the list to an essentia.array first (== numpy.array of floats)
        mfccs = essentia.array(mfccs).T
        melbands = essentia.array(melbands).T
        melbands_log = essentia.array(melbands_log).T

        # and plot
        imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
        plt.title("Mel band spectral energies in frames")
        show()

        imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
        plt.title("Log-normalized mel band spectral energies in frames")
        show()

        imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
        plt.title("MFCCs in frames")
        show()

        break








########################################################

audio_path = "../Audio/MIR-1K_mixture"

abs_audio_path = os.path.abspath(audio_path)

for filename in os.listdir(abs_audio_path):
    if filename.endswith(".wav"):
        feature_extraction(audio_path, filename)
    break
