import os
import hdf5plugin
import mido
import h5py
import json
import time
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
from tensorflow.keras.layers import Input, Resizing, Conv2D, BatchNormalization, Multiply
from keras import backend as K
import ray

ray.init(ignore_reinit_error=True)

############################################################

SAVE_MODEL = False
LOAD_MODEL = True
TRAINING = False
EVALUATE = False
EPOCHS = 10
COLD_PAUSE_BETWEEN_EPOCHS = False
TRAINING_DTYPE = tf.float16
SPLIT_SIZE = 256
BATCH_SIZE = 32
RESIZING_FILTER = 'bilinear'

############################################################

dataset_dir = "Datasets/"
checkpoint_dir = "Checkpoints/voas_cnn.keras"
midi_dir = "MIDI/"
zipname = dataset_dir + "SSCS_HDF5.zip"
sscs_dir = dataset_dir + "SSCS_HDF5/"

songs_dir = sscs_dir + "sscs/"
splitname = sscs_dir + "sscs_splits.json"

############################################################

def res_voas_cnn_model():
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER)(x_in)
    
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
    
    x = BatchNormalization()(x)

    ## "masking" original input with trained data

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER)(x)

    x = Multiply()([x, x_in])

    ## start four branches now

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
    y1 = tf.squeeze(y1, axis=-1, name='sop')

    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto',
                padding='same', activation='sigmoid')(x2b)
    y2 = tf.squeeze(y2, axis=-1, name='alt')

    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor',
                padding='same', activation='sigmoid')(x3b)
    y3 = tf.squeeze(y3, axis=-1, name='ten')

    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass',
                padding='same', activation='sigmoid')(x4b)
    y4 = tf.squeeze(y4, axis=-1, name='bas')

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='voasCNN')

    return model

############################################################

def voas_cnn_model():
    x_in = Input(shape=(360, SPLIT_SIZE, 1))
    
    x = BatchNormalization()(x_in)

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
    y1 = tf.squeeze(y1, axis=-1, name='sop')

    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto',
                padding='same', activation='sigmoid')(x2b)
    y2 = tf.squeeze(y2, axis=-1, name='alt')

    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor',
                padding='same', activation='sigmoid')(x3b)
    y3 = tf.squeeze(y3, axis=-1, name='ten')

    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass',
                padding='same', activation='sigmoid')(x4b)
    y4 = tf.squeeze(y4, axis=-1, name='bas')

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='voasCNN')

    return model

############################################################

def downsample_voas_cnn_model():
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

def initialize(save_model = True,
               load_model = False,
               training = True,
               evaluate = True,
               epochs = 10,
               cold_pause_between_epochs = False,
               training_dtype = tf.float16,
               split_size = 256,
               batch_size = 32,
               resizing_filter = 'bilinear'):
    SAVE_MODEL = save_model
    LOAD_MODEL = load_model
    TRAINING = training
    EVALUATE = evaluate
    EPOCHS = epochs
    COLD_PAUSE_BETWEEN_EPOCHS = cold_pause_between_epochs
    TRAINING_DTYPE = training_dtype
    SPLIT_SIZE = split_size
    BATCH_SIZE = batch_size
    RESIZING_FILTER = resizing_filter

    return

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

        batch_splits = np.array(list(map(self.__get_split__, tmp_idx, tmp_split)))

        splits = [tf.convert_to_tensor(batch_splits[:, i], dtype=self.training_dtype) for i in range(5)]

        return splits[0], (splits[1], splits[2], splits[3], splits[4]) # mix, (s, a, t, b)
    
    #-----------------------------------------------------------#
    
    def __get_split__(self, idx, split):

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
    split_str = str(split).lower()
    if(split_str == 'train' or split_str == 'validate' or split_str == 'test'):
        split_list = json.load(open(splitname, 'r'))[split_str]
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
    seq = get_sequence(split, 0, 2)
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

############################################################

def downsample_bins(voice):
    voice_0 = np.array(voice.T[0::5]).T
    voice_1 = np.array(voice.T[1::5]).T
    voice_2 = np.array(voice.T[2::5]).T
    voice_3 = np.array(voice.T[3::5]).T
    voice_4 = np.array(voice.T[4::5]).T

    voice_0 = voice_0.T[1:71].T
    voice_1 = voice_1.T[1:71].T
    voice_2 = voice_2.T[1:71].T
    voice_3 = voice_3.T[0:70].T
    voice_4 = voice_4.T[0:70].T

    voice_sums = voice_0 + voice_1 + voice_2 + voice_3 + voice_4

    def downsample_threshold(item):
        if item >= 1.0: return 1.0
        else: return 0.0

    vectorized_downsample_threshold = np.vectorize(downsample_threshold)

    voice_sums = vectorized_downsample_threshold(voice_sums)

    return voice_sums

############################################################

def create_midi(pr, write_path='./MIDI/midi_track.mid', ticks_per_beat=52,
                tempo=77, save_to_file=True, program=53, channel=0):
    
    if(not os.path.exists(midi_dir)):
        os.mkdir(midi_dir)

    def pr_to_list(pr):
        # List event = (pitch, velocity, time)
        T, N = pr.shape
        t_last = 0
        pr_tm1 = np.zeros(N)
        list_event = []
        for t in range(T):
            pr_t = pr[t]
            mask = (pr_t != pr_tm1)
            if(N == 360): range_step = 5
            else: range_step = 1
            if (mask).any():
                for n in range(0, N):
                    if mask[n]:
                        if(N <= 72):
                            pitch = 22 + n
                        else:
                            pitch = 21 + round(n/5)
                        if int(pr_t[n] * 127) >= 50:
                            velocity = 127
                        else:
                            velocity = 0
                        # Time is incremented since last event
                        t_event = t - t_last
                        t_last = t
                        list_event.append((pitch, velocity, t_event))
            pr_tm1 = pr_t
        list_event.append((0, 0, T - t_last))
        return list_event
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = mido.MidiFile()
    mid.ticks_per_beat = ticks_per_beat


    # Add a new track with the instrument name to the midi file
    track = mid.add_track("Voice Aah")
    # transform the matrix in a list of (pitch, velocity, time)
    events = pr_to_list(pr)
    #print(events)
    # Tempo
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
    track.append(mido.MetaMessage('channel_prefix', channel=channel))
    # Add the program_change
    #Choir Aahs = 53, Voice Oohs (or Doos) = 54, Synch Choir = 55
    track.append(mido.Message('program_change', program=program, channel=channel))

    # This list is required to shut down
    # notes that are on, intensity modified, then off only 1 time
    # Example :
    # (60,20,0)
    # (60,40,10)
    # (60,0,15)
    notes_on_list = []
    # Write events in the midi file
    for event in events:
        pitch, velocity, time = event
        if velocity == 0:
            # Get the channel
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=time, channel=channel))
            if(pitch in notes_on_list):
                notes_on_list.remove(pitch)
        else:
            if pitch in notes_on_list:
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=time, channel=channel))
                notes_on_list.remove(pitch)
                time = 0
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time, channel=channel))
            notes_on_list.append(pitch)
    if save_to_file:
        mid.save(write_path)
    return mid

############################################################

def song_to_midi(sop, alto, ten, bass, write_path='./MIDI/midi_mix.mid'):

    down_sop = downsample_bins(sop.T)
    down_alto = downsample_bins(alto.T)
    down_ten = downsample_bins(ten.T)
    down_bass = downsample_bins(bass.T)

    mid_sop = create_midi(down_sop, save_to_file=False, program=52, channel=0)
    mid_alto = create_midi(down_alto, save_to_file=False, program=53, channel=1)
    mid_ten = create_midi(down_ten, save_to_file=False, program=49, channel=2)
    mid_bass = create_midi(down_bass, save_to_file=False, program=50, channel=3)

    mid_mix = mido.MidiFile()
    mid_mix.ticks_per_beat=mid_sop.ticks_per_beat
    mid_mix.tracks = mid_sop.tracks + mid_alto.tracks + mid_ten.tracks + mid_bass.tracks
    mid_mix.save(write_path)

    return

############################################################

def songname_to_midi(songname, write_path='./MIDI/midi_mix.mid'):

    if(not os.path.exists(midi_dir)):
        os.mkdir(midi_dir)

    mix, satb = read_all_voices(songname)
    sop = satb[0].to_numpy().T
    alto = satb[1].to_numpy().T
    ten = satb[2].to_numpy().T
    bass = satb[3].to_numpy().T
    mix = mix.T

    down_sop = downsample_bins(sop)
    down_alto = downsample_bins(alto)
    down_ten = downsample_bins(ten)
    down_bass = downsample_bins(bass)

    mid_sop = create_midi(down_sop, save_to_file=False, program=52, channel=0)
    mid_alto = create_midi(down_alto, save_to_file=False, program=53, channel=1)
    mid_ten = create_midi(down_ten, save_to_file=False, program=49, channel=2)
    mid_bass = create_midi(down_bass, save_to_file=False, program=50, channel=3)

    mid_mix = mido.MidiFile()
    mid_mix.ticks_per_beat=mid_sop.ticks_per_beat
    mid_mix.tracks = mid_sop.tracks + mid_alto.tracks + mid_ten.tracks + mid_bass.tracks
    mid_mix.save(write_path)

    return

############################################################

def random_song_to_midi(write_path='./MIDI/midi_mix.mid'):
    song = pick_random_song()
    songname_to_midi(song, write_path)
    return

############################################################

def load_weights(model):
    if(LOAD_MODEL and os.path.exists(checkpoint_dir)):
        model.load_weights(checkpoint_dir)

############################################################

def train(model, ds_train, ds_val):

    save_cb = tf.keras.callbacks.ModelCheckpoint(   filepath=checkpoint_dir,
                                                    save_weights_only=True,
                                                    verbose=1
                                                )

    if(TRAINING):
            if(COLD_PAUSE_BETWEEN_EPOCHS):
                for epoch in range(EPOCHS):
                    if(SAVE_MODEL):
                        model.fit(ds_train,
                                    epochs=1,
                                    callbacks=[save_cb],
                                    validation_data=ds_val)
                    else:
                        model.fit(ds_train,
                                    epochs=1,
                                    validation_data=ds_val)
                    if(epoch < EPOCHS - 1):
                        print("Pausa para resfriar...")
                        time.sleep(120)
            else:
                if(SAVE_MODEL):
                    model.fit(ds_train,
                                epochs=EPOCHS,
                                callbacks=[save_cb],
                                validation_data=ds_val)
                else:
                    model.fit(ds_train,
                                epochs=EPOCHS,
                                validation_data=ds_val)


############################################################