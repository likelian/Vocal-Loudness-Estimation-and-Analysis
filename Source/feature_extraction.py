import soundfile as sf
import numpy as np
import json
import os


import librosa
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard

import tensorflow as tf
import tensorflow_hub as hub
model = hub.load('https://tfhub.dev/google/vggish/1')




def left_channel(audio):

    audio = np.array(audio.T[0], dtype=np.float32).T
    #audio = audio.astype(np.single)

    return audio


########################################################

def vggish(audio, sampleRate, model):
    """
    extract the mean of 128 VGG embeddings in 3s window
    """
    #sampleRate
    resample = essentia.standard.Resample(inputSampleRate=sampleRate, outputSampleRate=15600)
    audio_resampled = resample(audio)


    embedding_dict = {}
    vgg_mean_size = 0
    for idx in np.arange(10):
        embeddings = model(audio_resampled[idx*1560:]).numpy()

        embeddings_0 = embeddings[::3]
        embeddings_1 = embeddings[1:][::3]
        embeddings_2 = embeddings[2:][::3]

        max_length = max(embeddings_0.shape[0], embeddings_1.shape[0], embeddings_2.shape[0])

        embeddings_0 = np.pad(embeddings_0, ((0, max_length-embeddings_0.shape[0]+1), (0, 0)), 'constant')
        embeddings_1 = np.pad(embeddings_1, ((0, max_length-embeddings_1.shape[0]+1), (0, 0)), 'constant')
        embeddings_2 = np.pad(embeddings_2, ((0, max_length-embeddings_2.shape[0]+1), (0, 0)), 'constant')

        embeddings_mean_0 = np.mean([embeddings_0[:-1], embeddings_1[:-1], embeddings_2[:-1]], axis=0)
        embeddings_mean_1 = np.mean([embeddings_0[1:], embeddings_1[:-1], embeddings_2][:-1], axis=0)
        embeddings_mean_2 = np.mean([embeddings_0[1:], embeddings_1[1:], embeddings_2[:-1]], axis=0)

        embeddings_mean = np.zeros([embeddings_mean_0.shape[0]*3, 128])
        embeddings_mean[np.arange(embeddings_mean_0.shape[0])*3] = embeddings_mean_0
        embeddings_mean[np.arange(embeddings_mean_1.shape[0])*3+1] = embeddings_mean_1
        embeddings_mean[np.arange(embeddings_mean_2.shape[0])*3+2] = embeddings_mean_2


        embedding_dict[idx] = embeddings_mean
        vgg_mean_size += embeddings_mean.shape[0]

    vggish_mean = np.zeros([vgg_mean_size+50, 128])

    for idx in np.arange(10):
        vggish_mean[np.arange(embedding_dict[idx].shape[0])*10+idx] = embedding_dict[idx]

    return vggish_mean





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

    while current_time < audio_length -2.9999:

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

    while current_time <= audio_length -2.99:

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

def extract_features(audio_path, filename, feature_path = "../Features/MIR-1K/"):

    audio, sampleRate = sf.read(audio_path + "/" + filename)
    audio_stereo = audio
    audio = left_channel(audio)

    shortTermLUFS = shortTermLoudness(audio_stereo, sampleRate, HS=0.1)
    vggish_mean = vggish(audio, sampleRate, model)
    vggish_mean = vggish_mean[:shortTermLUFS.shape[0]-vggish_mean.shape[0]]
    mfcc_mean = MFCC(audio, sampleRate, audio_stereo)
    #spectral_centroid = extract_spectral_centroid(audio, sampleRate)


    if mfcc_mean.shape[0] != shortTermLUFS.shape[0]:
        mfcc_mean = mfcc_mean[:-1]
        print("mfcc")
        print(mfcc_mean.shape[0])
        print("shortTermLUFS")
        print(shortTermLUFS.shape[0])
        #quit()



    feature_dict = {}
    filename_noExt = filename[:-4]
    print(filename_noExt)


    feature_dict[filename_noExt+"_mfcc_mean"] = mfcc_mean.tolist()
    feature_dict[filename_noExt+"_shortLUFS"] = shortTermLUFS.tolist()
    feature_dict[filename_noExt+"_vggish_mean"] = vggish_mean.tolist()
    #feature_dict[filename_noExt+"_spectral_centroid"] = spectral_centroid.tolist()


    #feature_path = "../Features/"
    with open(feature_path + filename_noExt + "_features.json", 'w') as outfile:
        json.dump(feature_dict, outfile)


    #print(mfcc_mean.shape)



########################################################


def feature_extraction(audio_path = "../Audio/Mixture/MIR-1K_mixture", feature_path = "../Features/MIR-1K/"):

    abs_audio_path = os.path.abspath(audio_path)

    print(abs_audio_path)


    for filename in os.listdir(abs_audio_path):
        if filename.endswith(".wav"):
            extract_features(audio_path, filename, feature_path)
