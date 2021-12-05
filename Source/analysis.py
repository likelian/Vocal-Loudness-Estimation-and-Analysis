import helper
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; #sns.set()


analysis_path = "../Analysis/GTZAN"
abs_analysis_path = os.path.abspath(analysis_path)
vox_genre_dict = {}
vox_id_dict = {}
for filename in os.listdir(abs_analysis_path):

    if filename[-4:] != "json":
        continue

    idx_dot = filename.find('.')
    idx_under = filename.find('_')

    genre = filename[:idx_dot]
    id = filename[idx_dot+1:idx_under]

    if genre not in vox_genre_dict.keys():
        vox_genre_dict[genre] = np.zeros(100)

    f = open(analysis_path+"/"+filename)

    analysis_dict = json.load(f)

    for key in analysis_dict.keys():
        if "average_vox_shortTermLoudness" in key:
            first_zero_idx = np.where(vox_genre_dict[genre]==0)[0][0]
            vox_genre_dict[genre][first_zero_idx] = analysis_dict[key]
            vox_id_dict[genre+"."+id] = analysis_dict[key]





#remove jazz and classical
try:
    del vox_genre_dict['jazz']
    del vox_genre_dict['classical']
except:
    pass

vox_genre_pd = pd.DataFrame.from_dict(vox_genre_dict)


ax = sns.catplot(data=vox_genre_pd, kind="box")


ax.set(ylabel="Estimated Vocal Short-term LUFS in dB")


#plt.show()

plt.tight_layout(pad=1)

plt.savefig("../Analysis/GTZAN_results/" + "catplot" + '.png')


plt.close()

print(vox_genre_pd.mean())

#print(vox_id_dict)
