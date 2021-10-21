import ground_truth
import feature_extraction
#import machine_learning
import os


audio_path = "../Audio/MIR-1K"
mixture_audio_path = "../Audio/MIR-1K_mixture"
ground_truth_path = "../Ground_truth/"
feature_path = "../Features/"


feature_extraction(audio_path, feature_path)

ground_truth_generation(audio_path, ground_truth_path, mixture_audio_path)

os.system('python3 machine_learning.py')
