import soundfile as sf
import numpy as np
import sys
from scipy import signal
sys.path.append('/usr/local/lib/python3.8/site-packages')
import essentia
import essentia.standard
import json
import os


################################################################################################################

def ground_truth(audio_path, ground_truth_path, mixture_path, filename):

    data, sampleRate = sf.read(audio_path + "/" + filename)

    rand_dB = np.random.uniform(-6,0,1)

    rand_amp = 10**(rand_dB/20)

    acc = np.array([data.T[0]/2, data.T[0]/2], dtype=np.float32).T #Accompaniment
    vox = np.array([data.T[1]/2, data.T[1]/2], dtype=np.float32).T #Vocal
    str_rand = ''
    vox *= rand_amp
    str_rand = "_" + str(rand_dB)
    mix = (acc.T + vox.T).T #mono_sum


    acc_shortTermLoudness = shortTermLoudness(acc, SR=sampleRate)
    vox_shortTermLoudness = shortTermLoudness(vox, SR=sampleRate)
    mix_shortTermLoudness = shortTermLoudness(mix, SR=sampleRate)
    accREL_shortTermLoudness = acc_shortTermLoudness - mix_shortTermLoudness
    voxREL_shortTermLoudness = vox_shortTermLoudness - mix_shortTermLoudness


    acc_bandRMS = filter_bank(acc, sampleRate)
    vox_bandRMS = filter_bank(vox, sampleRate)
    mix_bandRMS = filter_bank(mix, sampleRate)
    acc_minus_vox_bandRMS =  acc_bandRMS - vox_bandRMS

    timeInSec = np.arange(acc_shortTermLoudness.size) * 0.1

    ground_truth = {}
    filename_noExt = filename[:-4]
    ground_truth[filename_noExt+"_timeInSec"] = timeInSec.tolist()
    ground_truth[filename_noExt+"_acc_shortTermLoudness"] = acc_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_vox_shortTermLoudness"] = vox_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_mix_shortTermLoudness"] = mix_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_accREL_shortTermLoudness"] = accREL_shortTermLoudness.tolist()
    ground_truth[filename_noExt+"_voxREL_shortTermLoudness"] = voxREL_shortTermLoudness.tolist()

    ground_truth[filename_noExt+"_acc_bandRMS"] = acc_bandRMS.tolist()
    ground_truth[filename_noExt+"_vox_bandRMS"] = vox_bandRMS.tolist()
    ground_truth[filename_noExt+"_mix_bandRMS"] = mix_bandRMS.tolist()
    ground_truth[filename_noExt+"_acc_minus_vox_bandRMS"] = acc_minus_vox_bandRMS.tolist()


    with open(ground_truth_path + filename_noExt + str_rand + "_ground_truth.json", 'w') as outfile:
        json.dump(ground_truth, outfile)

    filename = "mixture"  + "_" + filename[:-4] + str_rand + ".wav"
    mixture_path_filename = mixture_path + "/" + filename
    sf.write(mixture_path_filename, mix, sampleRate)


################################################################################################################

def shortTermLoudness(buffer, SR=44100, HS=0.1):
    """
    default sample rate: 44100Hz
    default hop size: 0.1s

    return: an array of shortTermLoudness in dB
    """
    LoudnessEBUR128 = essentia.standard.LoudnessEBUR128(sampleRate=SR, hopSize=HS)
    shortTermLoudness = LoudnessEBUR128(buffer)[1]
    return shortTermLoudness[:-1]



def filter_bank(audio, fs):
    """
    2nd order butterworth bandpass filter

    return:
        a numpy array of (10, window_num)
        window_num is the number of 3s sliding window of hop size of 0.1s
        the value in rms_dB_bank is the rms of each window after each filter
    """

    """
    import matplotlib.pyplot as plt
    #w, h = signal.sosfreqz(sos, worN=1500)
    #plt.semilogx(w, 20 * np.log10(abs(h)))
    #plt.title('Butterworth filter frequency response')
    #plt.xlabel('Frequency [radians / second]')
    #plt.ylabel('Amplitude [dB]')
    #plt.margins(0, 0.1)
    #plt.grid(which='both', axis='both')
    #plt.axvline(100, color='green') # cutoff frequency
    #plt.show()
    #plt.close()
    #sf.write("/Users/likelian/Desktop/bp.wav", filtered, fs)
    """

    audio = audio.T[0].T

    Nyquist = fs/2
    fcs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000] #ISO standard octave-band center frequencies

    window_num = int(np.ceil(((len(audio)/fs)-3)*10))
    rms_dB_bank = np.zeros((10, window_num))

    for idx, fc in enumerate(fcs):

        sos = signal.butter(2, [fc*2**(-0.5)/Nyquist, fc*2**0.5/Nyquist], 'bp', fs=fs, output='sos') #second order sections
        filtered = signal.sosfilt(sos, audio)

        sqarted = filtered**2
        blocked = np.array([sqarted[i:i+fs*3] for i in range(len(sqarted)-fs*3)][::int(fs/10)]) #3s overlaped blocks

        rms = np.sqrt(np.mean(blocked, axis=1))
        rms_dB = 20*np.log10(rms)
        rms_dB_bank[idx] = rms_dB

    return rms_dB_bank



################################################################################################################

def ground_truth_generation_MIR_1K(audio_path = "../Audio/MIR-1K",
                            ground_truth_path = "../Ground_truth/MIR-1K/",
                            mixture_path = "../Audio/Mixture/MIR-1K_mixture"):

    abs_audio_path = os.path.abspath(audio_path)

    for filename in os.listdir(abs_audio_path):
        if filename.endswith(".wav"):
            ground_truth(audio_path, ground_truth_path, mixture_path, filename)


################################################################################################################

def ground_truth_generation_MUSDB(audio_path = "../Audio/musdb18hq",
                            ground_truth_path = "../Ground_truth/musdb18hq",
                            mixture_path = "../Audio/Mixture/musdb18hq_mixture"):

    abs_audio_path = os.path.abspath(audio_path)

    for dir in os.listdir(abs_audio_path):
        stem_path = abs_audio_path+"/"+dir
        vox_path = stem_path+"/"+"vocals.wav"
        if not os.path.isfile(vox_path):
            continue
        original_mixture_path = stem_path+"/"+"mixture.wav"

        vox, sampleRate = sf.read(vox_path)
        mix_original, sampleRate = sf.read(original_mixture_path)

        acc = mix_original - vox

        vox_mono = vox.T[0]/2 + vox.T[1]/2
        acc_mono = acc.T[0]/2 + acc.T[1]/2


        rand_dB = np.random.uniform(-6,0,1)
        rand_amp = 10**(rand_dB/20)
        str_rand = ''
        vox_mono *= rand_amp
        str_rand = "_" + str(rand_dB)

        mix_mono = vox_mono + acc_mono

        vox = np.array([vox_mono, vox_mono], dtype=np.float32).T
        acc = np.array([acc_mono, acc_mono], dtype=np.float32).T
        mix = np.array([mix_mono, mix_mono], dtype=np.float32).T

        acc_shortTermLoudness = shortTermLoudness(acc, SR=sampleRate)
        vox_shortTermLoudness = shortTermLoudness(vox, SR=sampleRate)
        mix_shortTermLoudness = shortTermLoudness(mix, SR=sampleRate)
        accREL_shortTermLoudness = acc_shortTermLoudness - mix_shortTermLoudness
        voxREL_shortTermLoudness = vox_shortTermLoudness - mix_shortTermLoudness

        acc_bandRMS = filter_bank(acc, sampleRate)
        vox_bandRMS = filter_bank(vox, sampleRate)
        mix_bandRMS = filter_bank(mix, sampleRate)
        acc_minus_vox_bandRMS =  acc_bandRMS - vox_bandRMS


        timeInSec = np.arange(acc_shortTermLoudness.size) * 0.1

        ground_truth = {}
        filename = str(dir)
        ground_truth[filename+"_timeInSec"] = timeInSec.tolist()
        ground_truth[filename+"_acc_shortTermLoudness"] = acc_shortTermLoudness.tolist()
        ground_truth[filename+"_vox_shortTermLoudness"] = vox_shortTermLoudness.tolist()
        ground_truth[filename+"_mix_shortTermLoudness"] = mix_shortTermLoudness.tolist()
        ground_truth[filename+"_accREL_shortTermLoudness"] = accREL_shortTermLoudness.tolist()
        ground_truth[filename+"_voxREL_shortTermLoudness"] = voxREL_shortTermLoudness.tolist()

        ground_truth[filename+"_acc_bandRMS"] = acc_bandRMS.tolist()
        ground_truth[filename+"_vox_bandRMS"] = vox_bandRMS.tolist()
        ground_truth[filename+"_mix_bandRMS"] = mix_bandRMS.tolist()
        ground_truth[filename+"_acc_minus_vox_bandRMS"] = acc_minus_vox_bandRMS.tolist()


        with open(ground_truth_path +"/"+ filename + str_rand + "_ground_truth.json", 'w') as outfile:
            json.dump(ground_truth, outfile)

        mixture_path_filename = mixture_path+"/mixture_"+str(dir)+str_rand+".wav"
        sf.write(mixture_path_filename, mix, sampleRate)

################################################################################################################
