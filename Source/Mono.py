import soundfile as sf
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard as ess
import json


dir = "../Audio/"
filename = "amy_1"
wav = ".wav"
dir_filename = dir + filename + wav

data, sampleRate = sf.read(dir_filename)

acc = np.array([data.T[0]/2, data.T[0]/2], dtype=np.float32).T #Accompaniment
vox = np.array([data.T[1]/2, data.T[1]/2], dtype=np.float32).T #Vocal


########################################################

"""
from essentia.standard import *
w = Windowing(type = 'hann')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()
"""

def shortTermLoudness(buffer, SR=44100, HS=0.1):
    """
    default sample rate: 44100Hz
    default hop size: 0.1s

    return: an array of shortTermLoudness in dB
    """
    LoudnessEBUR128 = essentia.standard.LoudnessEBUR128(sampleRate=SR, hopSize=HS)
    shortTermLoudness = LoudnessEBUR128(buffer)[1]
    return shortTermLoudness


acc_shortTermLoudness = shortTermLoudness(acc, SR=sampleRate)
vox_shortTermLoudness = shortTermLoudness(vox, SR=sampleRate)


########################################################


ground_truth = {}
ground_truth[filename+"_acc_shortTermLoudness"] = acc_shortTermLoudness.tolist()
ground_truth[filename+"_vox_shortTermLoudness"] = vox_shortTermLoudness.tolist()


ground_truth_dir = "../JSON/ground_truth/"
with open(ground_truth_dir + 'ground_truth.txt', 'w') as outfile:
    json.dump(ground_truth, outfile)


mixture_Data = (acc.T + vox.T).T #mono_sum

mixture_dir_filename = dir + "mixture_" + filename + wav
sf.write(mixture_dir_filename, mixture_Data, sampleRate)
