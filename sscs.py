import os
import hdf5plugin
import h5py
import json
import zipfile
import requests
import psutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import Input, Resizing, Conv2D, BatchNormalization
from keras import backend as K
import ray

ray.init(ignore_reinit_error=True)

############################################################

EXECUTE_ON_COLAB = False
SAVE_MODEL = True
LOAD_MODEL = False
TRAINING = True
EVALUATE = True
EPOCHS = 10
COLD_PAUSE_BETWEEN_EPOCHS = False
TRAINING_DTYPE = tf.float16
SPLIT_SIZE = 256
BATCH_SIZE = 32
RESIZING_FILTER = 'bilinear'

############################################################

if(EXECUTE_ON_COLAB):
    dataset_dir = "/content/Datasets/"
    checkpoint_dir = "/content/drive/MyDrive/SSCS/Checkpoints/sscs.ckpt"
else:
    dataset_dir = "Datasets/"
    checkpoint_dir = "Checkpoints/voas_cnn.keras"
zipname = dataset_dir + "SSCS_HDF5.zip"
sscs_dir = dataset_dir + "SSCS_HDF5/"

songs_dir = sscs_dir + "sscs/"
splitname = sscs_dir + "sscs_splits.json"

############################################################

def voas_cnn_model():
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(216, int(SPLIT_SIZE/2), RESIZING_FILTER)(x_in)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(70, 3), padding="same",
        activation="relu", name="conv_harm_1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(70, 3), padding="same",
        activation="relu", name="conv_harm_2")(x)

    ## start four branches now

    x = BatchNormalization()(x)

    ## branch 1
    x1a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1a")(x)

    x1a = BatchNormalization()(x1a)

    x1b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1b")(x1a)

    ## branch 2
    x2a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2a")(x)

    x2a = BatchNormalization()(x2a)

    x2b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2b")(x2a)

    ## branch 3

    x3a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3a")(x)

    x3a = BatchNormalization()(x3a)

    x3b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3b")(x3a)

    x4a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4a")(x)

    x4a = BatchNormalization()(x4a)

    x4b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4b"
    )(x4a)


    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano',
                padding='same', activation='sigmoid')(x1b)
    y1 = Resizing(360, SPLIT_SIZE, RESIZING_FILTER)(y1)
    y1 = tf.squeeze(y1, axis=-1, name='sop')

    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto',
                padding='same', activation='sigmoid')(x2b)
    y2 = Resizing(360, SPLIT_SIZE, RESIZING_FILTER)(y2)
    y2 = tf.squeeze(y2, axis=-1, name='alt')

    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor',
                padding='same', activation='sigmoid')(x3b)
    y3 = Resizing(360, SPLIT_SIZE, RESIZING_FILTER)(y3)
    y3 = tf.squeeze(y3, axis=-1, name='ten')

    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass',
                padding='same', activation='sigmoid')(x4b)
    y4 = Resizing(360, SPLIT_SIZE, RESIZING_FILTER)(y4)
    y4 = tf.squeeze(y4, axis=-1, name='bas')

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='voasCNN')

    return model

############################################################

class SSCS_Sequence(tf.keras.utils.Sequence):
    
    #-----------------------------------------------------------#

    def __init__(self,
                 filenames,
                 batch_size=BATCH_SIZE,
                 split_size=SPLIT_SIZE,
                 training_dtype=TRAINING_DTYPE):

        if(isinstance(filenames, np.ndarray)):
            self.filenames = [f.decode('utf-8') for f in filenames.tolist()]
        else:
            self.filenames = filenames

        self.batch_size = batch_size
        self.batches_amount = 0
        self.splits_per_file = np.array([], dtype=np.intc)
        self.songs_dir = songs_dir
        self.split_size = split_size
        self.idx_get = np.array([], dtype=np.intc)
        self.split_get = np.array([], dtype=np.intc)
        self.training_dtype = training_dtype

        for file in self.filenames:

            file_access = f"{self.songs_dir}{file}.h5"
            f = h5py.File(file_access, 'r')
            file_shape = f['mix/table'].shape[0]
            df_batch_items = file_shape//self.split_size
            #if(file_shape/self.split_size > df_batch_items): df_batch_items += 1
            self.splits_per_file = np.append(self.splits_per_file, int(df_batch_items))
            tmp_idx_get = np.array([self.filenames.index(file) for i in range(df_batch_items)], dtype=np.intc)
            tmp_split_get = np.array([i for i in range(df_batch_items)], dtype=np.intc)
            self.idx_get = np.append(self.idx_get, tmp_idx_get)
            self.split_get = np.append(self.split_get, tmp_split_get)
            f.close()
        
        self.batches_amount = self.split_get.shape[0]//self.batch_size
        if self.batches_amount < self.split_get.shape[0]/self.batch_size: 
            self.batches_amount += 1

        self.idx_get = np.resize(self.idx_get, self.batches_amount * self.batch_size)
        self.idx_get = np.reshape(self.idx_get, (-1, self.batch_size))

        self.split_get = np.resize(self.split_get, self.batches_amount * self.batch_size)
        self.split_get = np.reshape(self.split_get, (-1, self.batch_size))
     
    #-----------------------------------------------------------#

    def __len__(self):

        return self.batches_amount
    
    #-----------------------------------------------------------#

    def __getitem__(self, idx):

        tmp_idx = self.idx_get[idx]
        tmp_split = self.split_get[idx]

        batch_splits = np.array(list(map(self.get_split, tmp_idx, tmp_split)))

        splits = [tf.convert_to_tensor(batch_splits[:, i], dtype=self.training_dtype) for i in range(5)]

        return splits[0], (splits[1], splits[2], splits[3], splits[4]) # mix, (s, a, t, b)
    
    #-----------------------------------------------------------#
    
    def get_split(self, idx, split):

        file_access = f"{self.songs_dir}{self.filenames[idx]}.h5"
        data_min = split * self.split_size
        data_max = data_min + self.split_size
        voices = ['mix', 'soprano', 'alto', 'tenor', 'bass']

        def read_split(voice):

            f = h5py.File(file_access, 'r')

            data = np.transpose(np.array([line[1] for line in f[voice + "/table"][data_min:data_max]]))
            data = data.reshape((data.shape[0], data.shape[1], 1))

            f.close()

            return data

        splits = list(map(read_split, voices))

        return splits # mix, soprano, alto, tenor, bass
    
    #-----------------------------------------------------------#

    def get_splits_per_file(self):
        
        return self.splits_per_file

############################################################

def dl_script(url, fname):
    
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    downloaded_size = 0
    with open(fname, 'wb') as file:
        for data in resp.iter_content(chunk_size=max(4096, int(total/10000))):
            size = file.write(data)
            downloaded_size += size
            percent = min(downloaded_size/total, 1.0)
            print(f"\r{percent:.2%} downloaded", end='')
            
    print()

############################################################

def download():
    
    if(not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)
   
    if(not os.path.exists(zipname)):
        print("Downloading SSCS Dataset...")
        url = "https://github.com/Xornotor/SSCS_HDF5/releases/download/v1.0/SSCS_HDF5.zip"
        dl_script(url, zipname)
    else:
        print("SSCS Dataset found.")

    if(not os.path.exists(sscs_dir)):
        print("Extracting SSCS Dataset...")
        with zipfile.ZipFile(zipname) as zf:
            os.mkdir(sscs_dir)
            zf.extractall(path=sscs_dir)
    else:
        print("SSCS Dataset already extracted.")
    
    print("Done.")

############################################################

def get_split(split='train'):
    
    if(split.lower() == 'train' or split.lower() == 'validate' or
       split.lower() == 'test'):
        split_list = json.load(open(splitname, 'r'))[split.lower()]
        return split_list
    else:
        raise NameError("Split should be 'train', 'validate' or 'test'.")
    
############################################################

def pick_songlist(first=0, amount=5, split='train'):
    
    songnames = get_split(split)
    return songnames[first:first+amount]

############################################################

def pick_random_song(split='train'):
    
    songnames = get_split(split)
    rng = np.random.randint(0, len(songnames))
    return songnames[rng]

############################################################

def pick_multiple_random_songs(amount, split='train'):
    
    return [pick_random_song() for i in range(amount)]

############################################################

@ray.remote
def read_voice(name, voice):

    if  (voice != 'mix' and \
        voice != 'soprano' and \
        voice != 'alto' and \
        voice != 'tenor' and \
        voice != 'bass'):
        raise NameError("Specify voice with 'soprano', 'alto', \
                        'tenor', 'bass' or 'mix'.")
    
    filename = songs_dir + name + ".h5"
    return pd.read_hdf(filename, voice).T

############################################################

def read_all_voices(name):
    
    voices = ['mix', 'soprano', 'alto', 'tenor', 'bass']
    data_access = [read_voice.remote(name, voice) for voice in voices]
    df_voices = ray.get(data_access)
    mix = df_voices[0]
    satb = df_voices[1:]
    return mix, satb

############################################################

@ray.remote
def split_and_reshape(df, split_size):
    
    split_arr = np.array_split(df, df.shape[1]/split_size, axis=1)
    split_arr = np.array([i.iloc[:, :split_size] for i in split_arr])
    return split_arr

############################################################

@ray.remote
def read_all_voice_splits(name, split_size):
    
    mix_raw, satb_raw = read_all_voices(name)
    df_voices = satb_raw
    df_voices.insert(0, mix_raw)
    voice_splits = [split_and_reshape.remote(df, split_size) for df in df_voices]
    mix_splits = ray.get(voice_splits)[0]
    s_splits = ray.get(voice_splits)[1]
    a_splits = ray.get(voice_splits)[2]
    t_splits = ray.get(voice_splits)[3]
    b_splits = ray.get(voice_splits)[4]
    return mix_splits, s_splits, a_splits, t_splits, b_splits

############################################################

def read_multiple_songs_splits(split_size, first=0, amount=5, split='train'):
    
    songlist = pick_songlist(first, amount, split)
    split_access = [read_all_voice_splits.remote(song, split_size) \
                    for song in songlist]
    split_list = ray.get(split_access)

    mix_list = [split_list[i][0] for i in range(amount)]
    s_list = [split_list[i][1] for i in range(amount)]
    a_list = [split_list[i][2] for i in range(amount)]
    t_list = [split_list[i][3] for i in range(amount)]
    b_list = [split_list[i][4] for i in range(amount)]

    mix_splits = np.concatenate(mix_list, axis=0)
    s_splits = np.concatenate(s_list, axis=0)
    a_splits = np.concatenate(a_list, axis=0)
    t_splits = np.concatenate(t_list, axis=0)
    b_splits = np.concatenate(b_list, axis=0)

    input = mix_splits
    outputs = (s_splits, a_splits, t_splits, b_splits)
    
    return input, outputs

############################################################

def plot(dataframe):

    aspect_ratio = (3/8)*dataframe.shape[1]/dataframe.shape[0]
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(dataframe, interpolation='nearest', aspect=aspect_ratio,
        cmap = mpl.colormaps['BuPu'])
    ax.invert_yaxis()
    plt.show()

############################################################

def plot_random(voice, split='train'):
    
    random_song = pick_random_song(split)
    plot(ray.get(read_voice.remote(random_song, voice)))

############################################################

def get_sequence(split='train', start_index=0, end_index=1000):
    return SSCS_Sequence(get_split(split)[start_index:end_index])

############################################################

def get_dataset(split='train', start_index=0, end_index=1000):
    seq = get_sequence(0, 2)
    mix_test, satb_test = seq.__getitem__(0)
    ds_spec = tf.TensorSpec(shape=mix_test.shape, dtype=TRAINING_DTYPE)
    signature = (ds_spec, (ds_spec, ds_spec, ds_spec, ds_spec))
    if(split=='train'):
        ds = tf.data.Dataset.from_generator(SSCS_Sequence,
                                    args = [get_split(split)[start_index:end_index]],
                                    output_signature=signature
                                    ).shuffle(10, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(SSCS_Sequence,
                                    args = [get_split(split)[start_index:end_index]],
                                    output_signature=signature
                                    ).prefetch(tf.data.AUTOTUNE)
    return ds