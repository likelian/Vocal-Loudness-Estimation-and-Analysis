import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import json


dir = "../Audio/"
filename = "amy_1"
wav = ".wav"
dir_filename = dir + filename + wav

data, samplerate = sf.read(dir_filename)
#length = data.shape[0] / samplerate

acc = (data.T[0]/2).T #Accompaniment
vox = (data.T[1]/2).T #Vocal


meter = pyln.Meter(samplerate) # create BS.1770 meter

acc_integratedLUFS = meter.integrated_loudness(acc)
vox_integratedLUFS = meter.integrated_loudness(vox)

integratedLUFS = {"acc_integratedLUFS": acc_integratedLUFS,
                  "vox_integratedLUFS": vox_integratedLUFS}

ground_truth = {}
ground_truth[filename] = integratedLUFS



ground_truth_dir = "../JSON/ground_truth/"
with open(ground_truth_dir + 'ground_truth.txt', 'w') as outfile:
    json.dump(ground_truth, outfile)



mono_Data = (acc.T + vox.T).T #mono_sum

mono_dir_filename = dir + "mono_" + filename + wav
sf.write(mono_dir_filename, mono_Data, samplerate)
