import os
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from urllib.request import urlopen
import shutil

pathname = "Datasets"
zipname = pathname + "/SynthSalienceChoralSet_v1.zip"

def download():
    if(not os.path.exists(pathname)):
        os.mkdir(pathname)
    if(not os.path.exists(zipname)):
        url = "https://zenodo.org/record/6534429/files/SynthSalienceChoralSet_v1.zip?download=1"
        with urlopen(url) as response, open(zipname, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

def read_metadata():
    with zipfile.ZipFile(zipname) as zf:
        with zf.open('sscs_metadata.csv') as f:
            df = pd.read_csv(f)
            return df

def read_voice(name, voice):
    filename = 'sscs/' + name + "_"
    if(voice.upper() == 'S' or voice.upper() == 'A' or \
       voice.upper() == 'T' or voice.upper() == 'B'):
        filename = filename + voice.upper()
    elif(voice.lower() == 'mix'):
        filename = filename + voice.lower()
    else:
        raise NameError("Specify voice with 'S', 'A', 'T', 'B' or 'mix'.")
    filename = filename + ".csv"
    with zipfile.ZipFile(zipname) as zf:
        with zf.open(filename) as f:
            df = pd.read_csv(f)
    return df

def pick_random_song():
    df_metadata = read_metadata()
    rng = np.random.randint(0, df_metadata.shape[0])
    return df_metadata.get("Song name")[rng]

def read_all_voices(name):
    df_mix = read_voice(name, 'mix')
    df_s = read_voice(name, 'S')
    df_a = read_voice(name, 'A')
    df_t = read_voice(name, 'T')
    df_b = read_voice(name, 'B')
    return df_mix, df_s, df_a, df_t, df_b

def plot(dataframe):
  aspect_ratio = (3/8)*dataframe.shape[1]/dataframe.shape[0]
  fig, ax = plt.subplots(figsize=(13, 7))
  im = ax.imshow(dataframe, interpolation='nearest', aspect=aspect_ratio)
  plt.show()