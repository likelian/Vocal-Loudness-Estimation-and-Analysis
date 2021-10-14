import soundfile as sf
import numpy as np
import json
import os

import librosa
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard




def left_channel(audio):

    audio = np.array(audio.T[0], dtype=np.float32).T
    #audio = audio.astype(np.single)

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
        else:
            xb = newBlock

        timeInSample = np.append(timeInSample, currentTimeInSample)
        currentTimeInSample += hopSize

    timeInSec = timeInSample / fs
    return xb, timeInSec


########################################################



def MFCC(audio, sampleRate, audio_stereo):
    """
    extract the 20 MFFC mean values in 3s window with a hop size of 0.1s

    return:
        mfcc_mean: a matrix of Nx20
    """

    audio_length = audio.size/sampleRate

    hop_size = 512

    mfccs = librosa.feature.mfcc(audio, sr=sampleRate, hop_length=hop_size) #window 2048

    timeInSec = np.arange(mfccs.shape[1]) * hop_size / sampleRate

    mfcc_mean = None
    current_time = 0.0

    mfccs_T = mfccs.T

    #print(audio_length - np.max(timeInSec))

    while current_time <= audio_length -3 + (0/sampleRate):

        idx_min = np.searchsorted(timeInSec, current_time, 'right')
        idx_max = np.searchsorted(timeInSec, current_time+3.0, 'left') - 1

        current_time += 0.1

        if mfcc_mean is None:
            mfcc_mean = np.mean(mfccs_T[idx_min:idx_max], axis=0)[:,np.newaxis].T

        else:
            mfcc_mean = np.concatenate([mfcc_mean, np.mean(mfccs_T[idx_min:idx_max],axis=0)[:,np.newaxis].T], axis=0)


    return mfcc_mean


########################################################



def shortTermLoudness(buffer, SR=44100, HS=0.1):
    """
    default sample rate: 44100Hz
    default hop size: 0.1s

    return: an array of shortTermLoudness in dB
    """
    LoudnessEBUR128 = essentia.standard.LoudnessEBUR128(sampleRate=SR, hopSize=HS)
    shortTermLoudness = LoudnessEBUR128(buffer)[1][:-1]

    return shortTermLoudness



########################################################

#spectral_centroid

########################################################

#spectral_flatness

########################################################

#spectral_rolloffxe

########################################################

#poly_features

########################################################

#zero_crossing_rate

########################################################

def feature_extraction(audio_path, filename):

    audio, sampleRate = sf.read(audio_path + "/" + filename)
    audio_stereo = audio
    audio = left_channel(audio)


    mfcc_mean = MFCC(audio, sampleRate, audio_stereo)
    shortTermLoudness = shortTermLoudness(audio_stereo, sampleRate, HS=0.1)



    feature_dict = {}
    filename_noExt = filename[8:-4]

    #print(filename_noExt)

    feature_dict[filename_noExt+"_mfcc_mean"] = mfcc_mean.tolist()
    feature_dict[filename_noExt+"_shortLUFS"] = shortTermLoudness.tolist()


    feature_path = "../Features/"
    with open(feature_path + filename_noExt + "_features.json", 'w') as outfile:
        json.dump(feature_dict, outfile)


    #print(mfcc_mean.shape)




########################################################

audio_path = "../Audio/MIR-1K_mixture"

abs_audio_path = os.path.abspath(audio_path)

counter = 0
for filename in os.listdir(abs_audio_path):
    if filename.endswith(".wav"):
        feature_extraction(audio_path, filename)

    #counter += 1
    #if counter >= 10: break