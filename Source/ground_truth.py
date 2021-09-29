import soundfile as sf
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard as ess
import json
import os


########################################################

def ground_truth(audio_path, filename):

    data, sampleRate = sf.read(audio_path + "/" + filename)

    acc = np.array([data.T[0]/2, data.T[0]/2], dtype=np.float32).T #Accompaniment
    vox = np.array([data.T[1]/2, data.T[1]/2], dtype=np.float32).T #Vocal
    mix = (acc.T + vox.T).T #mono_sum

    acc_shortTermLoudness = shortTermLoudness(acc, SR=sampleRate)
    vox_shortTermLoudness = shortTermLoudness(vox, SR=sampleRate)
    mix_shortTermLoudness = shortTermLoudness(mix, SR=sampleRate)
    accREL_shortTermLoudness = acc_shortTermLoudness - mix_shortTermLoudness
    voxREL_shortTermLoudness = vox_shortTermLoudness - mix_shortTermLoudness

    ground_truth = {}
    filename_noExt = filename[:-4]
    ground_truth[filename_noExt+"_acc_shortTermLoudness"] = acc_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_vox_shortTermLoudness"] = vox_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_mix_shortTermLoudness"] = mix_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_accREL_shortTermLoudness"] = accREL_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_voxREL_shortTermLoudness"] = voxREL_shortTermLoudness.tolist()

    ground_truth_path = "../Ground_truth/"
    with open(ground_truth_path + filename_noExt + "_ground_truth.json", 'w') as outfile:
        json.dump(ground_truth, outfile)

    mixture_path = "../Audio/MIR-1K_mixture"
    #mix(acc, vox, sampleRate, mixture_path, filename)
    filename  = "mixture" + "_" + filename
    mixture_path_filename = mixture_path + "/" + filename
    sf.write(mixture_path_filename, mix, sampleRate)

########################################################
"""
def mix(acc, vox, sampleRate, mixture_path, filename):
    mixture_Data = (acc.T + vox.T).T #mono_sum
    filename  = "mixture" + "_" + filename
    mixture_path_filename = mixture_path + "/" + filename
    sf.write(mixture_path_filename, mixture_Data, sampleRate)
    """

########################################################

def shortTermLoudness(buffer, SR=44100, HS=0.1):
    """
    default sample rate: 44100Hz
    default hop size: 0.1s

    return: an array of shortTermLoudness in dB
    """
    LoudnessEBUR128 = essentia.standard.LoudnessEBUR128(sampleRate=SR, hopSize=HS)
    shortTermLoudness = LoudnessEBUR128(buffer)[1]
    return shortTermLoudness


########################################################


audio_path = "../Audio/MIR-1K"

abs_audio_path = os.path.abspath(audio_path)

for filename in os.listdir(abs_audio_path):
    if filename.endswith(".wav"):
        ground_truth(audio_path, filename)

########################################################
