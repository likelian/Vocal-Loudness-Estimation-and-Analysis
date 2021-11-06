import ground_truth
import feature_extraction
#import machine_learning
import os


audio_path = "../Audio/MIR-1K"
mixture_audio_path = "../Audio/Mixture/MIR-1K_mixture"
ground_truth_path = "../Ground_truth/MIR-1K"
feature_path = "../Features/MIR-1K/"


#counter = 0
#while True:
#    ground_truth.ground_truth_generation(audio_path, ground_truth_path, mixture_audio_path)
#    counter += 1
#    if counter >= 2: break;

feature_extraction.feature_extraction(mixture_audio_path, feature_path)

#os.system('python3 machine_learning.py')
