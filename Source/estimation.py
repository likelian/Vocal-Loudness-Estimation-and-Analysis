import numpy as np
#import json
import os
#from sklearn.pipeline import make_pipeline
#from sklearn.multioutput import RegressorChain
import time
import helper
import matplotlib.pyplot as plt
import pickle
import feature_extraction
import librosa
import soundfile as sf
import json





def estimate(chain, X_test):

    scalerfile = '../Model/scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    X_test = scaler.transform(X_test)

    y_pred = chain.predict(X_test)

    #post-processing
    y_pred = np.where(y_pred <= 0.00001, 0.000001, y_pred)
    y_pred = np.where(y_pred >= 1, 1, y_pred)

    y_pred = 20 * np.log10(y_pred) #convert amplitude to dB
    y_pred = np.nan_to_num(y_pred, nan=-15, posinf=0, neginf=-15)


    return y_pred



# load
model_location = '../model/svr.pkl'
with open(model_location, 'rb') as f:
    chain = pickle.load(f)



audio_path = "../Audio/fma_small"
abs_audio_path = os.path.abspath(audio_path)

feature_path = "../Features/fma_small/"
abs_feature_path = os.path.abspath(feature_path) + "/"
if not os.path.exists(abs_feature_path):
    os.makedirs(abs_feature_path)

analysis_path = "../Analysis/fma_small"
abs_analysis_path = os.path.abspath(analysis_path)


for folder in [x[0] for x in os.walk(abs_audio_path)][1:]:

    for filename in os.listdir(folder):

        if filename[-4:] != ".mp3":
            continue

        print(filename[:-4])

        if filename[:-4]+"_analysis.json" in os.listdir(abs_analysis_path):
            print(filename[:-4]+"_analysis.json" )
            continue

        try:
            audio, sampleRate = librosa.load(folder + "/" + filename)
        except:
            continue

        stereo = np.array([audio.T, audio.T]).T

        wav_name = filename[:-4]+".wav"
        sf.write(folder + "/" + wav_name, stereo, sampleRate)

        try:
            feature_extraction.extract_features(folder, wav_name, feature_path = abs_feature_path)
        except:
            continue

        f_feature = open(abs_feature_path+"/"+ wav_name[:-4]+'_features.json')

        feature_dict = json.load(f_feature)

        current_features = None
        for key in list(feature_dict.keys()):
            if current_features is None:
                current_features = np.array(feature_dict[key])
            else:
                current_features = np.concatenate([current_features, np.array(feature_dict[key])], axis=1)


        y_pred = estimate(chain, current_features)

        pred_nan_idx = np.argwhere(y_pred.T[1]<=-14.999)
        y_pred_nan = np.copy(y_pred)
        y_pred_nan[pred_nan_idx] = np.nan
        y_pred_mean = np.nanmean(y_pred, axis=0)


        analysis_dict = {}
        analysis_dict[wav_name[:-4]+"_average_acc_shortTermLoudness"] = y_pred_mean[0]
        analysis_dict[wav_name[:-4]+"_average_vox_shortTermLoudness"] = y_pred_mean[1]
        analysis_dict[wav_name[:-4]+"_average_vox_band"] = y_pred_mean[2:].tolist()
        analysis_dict[wav_name[:-4]+"_estimation_array"] = y_pred.tolist()

        with open(abs_analysis_path + "/"+ wav_name[:-4] + "_analysis.json", 'w') as outfile:
            json.dump(analysis_dict, outfile)




        #check the dataset, skip no vocal
        #write y_pred_mean and y_pred along with filename to json
