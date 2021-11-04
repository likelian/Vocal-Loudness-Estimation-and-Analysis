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

    return: a Nx1 matrix of shortTermLoudness in dB
    """
    LoudnessEBUR128 = essentia.standard.LoudnessEBUR128(sampleRate=SR, hopSize=HS)
    shortTermLoudness = LoudnessEBUR128(buffer)[1][:-1].T[:, np.newaxis]


    return shortTermLoudness



########################################################

#spectral_centroid
def extract_spectral_centroid(audio, sampleRate=44100):
    """
    default sample rate: 44100Hz
    default hop size: 0.1s

    return: a Nx1 matrix of spectral_centroid_mean in Hz
    """


    audio_length = audio.size/sampleRate

    hop_size = 512
    spectral_centroid = librosa.feature.spectral_centroid(audio, sampleRate, hop_length=hop_size)

    timeInSec = np.arange(spectral_centroid.size) * hop_size / sampleRate


    spectral_centroid_mean = None
    current_time = 0.0

    while current_time <= audio_length -3 + (0/sampleRate):

        idx_min = np.searchsorted(timeInSec, current_time, 'right')
        idx_max = np.searchsorted(timeInSec, current_time+3.0, 'left') - 1

        current_time += 0.1

        if spectral_centroid_mean is None:
            spectral_centroid_mean = np.mean(spectral_centroid[idx_min:idx_max])

        else:
            spectral_centroid_mean = np.append(spectral_centroid_mean, np.mean(spectral_centroid[idx_min:idx_max]))

    spectral_centroid_mean = spectral_centroid_mean.T[:, np.newaxis]



    return spectral_centroid_mean



########################################################

#spectral_flatness

########################################################

#spectral_rolloffxe

########################################################

#poly_features

########################################################

#zero_crossing_rate

########################################################

def extract_features(audio_path, filename, feature_path = "../Features/"):

    audio, sampleRate = sf.read(audio_path + "/" + filename)
    audio_stereo = audio
    audio = left_channel(audio)


    mfcc_mean = MFCC(audio, sampleRate, audio_stereo)
    shortTermLUFS = shortTermLoudness(audio_stereo, sampleRate, HS=0.1)
    spectral_centroid = extract_spectral_centroid(audio, sampleRate)


    feature_dict = {}
    filename_noExt = filename[:-4]
    print(filename_noExt)


    feature_dict[filename_noExt+"_mfcc_mean"] = mfcc_mean.tolist()
    feature_dict[filename_noExt+"_shortLUFS"] = shortTermLUFS.tolist()
    feature_dict[filename_noExt+"spectral_centroid"] = spectral_centroid.tolist()


    feature_path = "../Features/"
    with open(feature_path + filename_noExt + "_features.json", 'w') as outfile:
        json.dump(feature_dict, outfile)


    #print(mfcc_mean.shape)




########################################################


def feature_extraction(audio_path = "../Audio/MIR-1K_mixture", feature_path = "../Features/"):

    abs_audio_path = os.path.abspath(audio_path)

    print(abs_audio_path)


    for filename in os.listdir(abs_audio_path):
        if filename.endswith(".wav"):
            extract_features(audio_path, filename, feature_path)
