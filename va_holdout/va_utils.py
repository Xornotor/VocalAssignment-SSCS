'''
va_utils.py
Developed by André Paiva - 2023

The va_utils module was created to provide functions in order to
facilitate the implementation, management, training and metrics
evaluation (with SSCS test subset) of the proposed Voice Assignment
models. There are also functions to convert the isolated freq-bin
representations into a MIDI file.
'''

import os
import hdf5plugin
import h5py
import mido
import json
import time
import zipfile
import requests
import librosa
import mir_eval
import psutil
import numpy as np
import pandas as pd
import datetime
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.layers import Input, Resizing, Conv2D, BatchNormalization, Multiply
from keras import backend as K
import logging
import ray
import va_plots

ray.init(configure_logging=True, logging_level=logging.ERROR,
         num_cpus=6, num_gpus=1, ignore_reinit_error=True)
os.system("load_ext tensorboard")

############################################################

EPOCHS = 10
TRAINING_DTYPE = tf.float16
SPLIT_SIZE = 256
BATCH_SIZE = 24
LEARNING_RATE = 2e-3
RESIZING_FILTER = 'bilinear'

############################################################

dataset_dir = "Datasets/"
checkpoint_dir = "Checkpoints/voas_cnn.keras"
log_dir = "Logs/"
midi_dir = "MIDI/"
zipname = dataset_dir + "SSCS_HDF5.zip"
sscs_dir = dataset_dir + "SSCS_HDF5/"

songs_dir = sscs_dir + "sscs/"
splitname = sscs_dir + "sscs_splits.json"


############################################################

def voas_cnn_model(l_rate = LEARNING_RATE):
    """Loads VoasCNN compiled model. 

    Parameters
    ----------
    ``l_rate`` : Float
        Learning rate for training.

    Returns
    -------
    ``model`` : tf.Model
        Compiled VoasCNN model.
    """
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

    model = Model(inputs=x_in, outputs=out, name='VoasCNN')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    return model

############################################################

def downsample_voas_cnn_model(l_rate = LEARNING_RATE):
    """Loads DownsampleVoasCNN compiled model. 

    Parameters
    ----------
    ``l_rate`` : Float
        Learning rate for training.

    Returns
    -------
    ``model`` : tf.Model
        Compiled DownsampleVoasCNN model.
    """
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

    model = Model(inputs=x_in, outputs=out, name='DownsampleVoasCNN')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    return model

############################################################

def downsample_voas_cnn_v2_model(l_rate = LEARNING_RATE):
    """Loads DownsampleVoasCNNv2 compiled model. 

    Parameters
    ----------
    ``l_rate`` : Float
        Learning rate for training.

    Returns
    -------
    ``model`` : tf.Model
        Compiled DownsampleVoasCNNv2 model.
    """
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER,
                 name="downscale")(x_in)
    
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

    ## Resize to original resolution

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER,
                 name="upscale")(x)

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

    model = Model(inputs=x_in, outputs=out, name='DownsampleVoasCNNv2')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    return model

############################################################

def mask_voas_cnn_model(l_rate = LEARNING_RATE):
    """Loads MaskVoasCNN compiled model. 

    Parameters
    ----------
    ``l_rate`` : Float
        Learning rate for training.

    Returns
    -------
    ``model`` : tf.Model
        Compiled MaskVoasCNN model.
    """
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER,
                 name="downscale")(x_in)
    
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

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER,
                 name="upscale")(x)

    x = Multiply(name="multiply_mask")([x, x_in])

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

    model = Model(inputs=x_in, outputs=out, name='MaskVoasCNN')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    return model

############################################################

def mask_voas_cnn_v2_model(l_rate = LEARNING_RATE):
    """Loads MaskVoasCNNv2 compiled model. 

    Parameters
    ----------
    ``l_rate`` : Float
        Learning rate for training.

    Returns
    -------
    ``model`` : tf.Model
        Compiled MaskVoasCNNv2 model.
    """
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER,
                 name="downscale")(x_in)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(48, 3), padding="same",
        activation="relu", name="conv_harm_1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(48, 3), padding="same",
        activation="relu", name="conv_harm_2")(x)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=1, padding="same",
        activation="sigmoid", name="conv_sigmoid_before_mask")(x)

    ## "masking" original input with trained data

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER,
                 name="upscale")(x)

    x = Multiply(name="multiply_mask")([x, x_in])

    x = BatchNormalization()(x)

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

    model = Model(inputs=x_in, outputs=out, name='MaskVoasCNNv2')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    return model

############################################################

class SSCS_Sequence(tf.keras.utils.Sequence):
    """Sequence iterator to access SSCS Dataset."""
    
    #-----------------------------------------------------------#

    def __init__(self,
                 filenames,
                 batch_size=BATCH_SIZE,
                 split_size=SPLIT_SIZE,
                 training_dtype=TRAINING_DTYPE):

        """SSCS_Sequence constructor.

        Parameters
        ----------
        ``filenames`` : list of str
            List of song filenames.
        ``batch_size`` : int
            Literally batch size of the iterator objects.
        ``split_size`` : int
            Length (in time bins) for each element in batch.
        ``training_dtype`` : dtype
            Dtype for the numeric values.
        """

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

        """Returns number of batches of the iterator."""

        return self.batches_amount
    
    #-----------------------------------------------------------#

    def __getitem__(self, idx):
        """Gets an iterator item.

        Parameters
        ----------
        ``idx`` : int
            Index of the item.

        Returns
        ----------
        ``splits[0]`` : tf.Tensor
            Tensor for the mix splits.
        ``(splits[1], splits[2], splits[3], splits[4])`` : Tuple of tf.Tensors
            Tensors for the Soprano, Alto, Tenor and Bass splits.
        """

        tmp_idx = self.idx_get[idx]
        tmp_split = self.split_get[idx]

        batch_splits = np.array(list(map(self.__get_split__, tmp_idx, tmp_split)))

        splits = [tf.convert_to_tensor(batch_splits[:, i], dtype=self.training_dtype) for i in range(5)]

        return splits[0], (splits[1], splits[2], splits[3], splits[4]) # mix, (s, a, t, b)
    
    #-----------------------------------------------------------#
    
    def __get_split__(self, idx, split):

        """Gets a song split.

        Parameters
        ----------
        ``idx`` : int
            Index of the item.
        ``split`` : int
            Index of the split.

        Returns
        -------
        ``splits`` : list of ndarray
            List containing Mix, Soprano, Alto, Tenor and Bass splits, in this order.
        """

        file_access = f"{self.songs_dir}{self.filenames[idx]}.h5"
        data_min = split * self.split_size
        data_max = data_min + self.split_size
        voices = ['mix', 'soprano', 'alto', 'tenor', 'bass']

        def read_split(voice):

            """Gets a voice split.

            Parameters
            ----------
            ``voice`` : str
                Voice from which the split is read (mix, soprano, alto, tenor, bass)

            Returns
            -------
            ``data`` : ndarray
                Split from the desired voice
            """

            f = h5py.File(file_access, 'r')

            data = np.transpose(np.array([line[1] for line in f[voice + "/table"][data_min:data_max]]))
            data = data.reshape((data.shape[0], data.shape[1], 1))

            f.close()

            return data

        splits = list(map(read_split, voices))

        return splits # mix, soprano, alto, tenor, bass
    
    #-----------------------------------------------------------#

    def get_splits_per_file(self):
        
        """Returns the number of splits from each file on the iterator."""
        
        return self.splits_per_file

############################################################

def dl_script(url, fname):

    """Download file script.

    Parameters
    ----------
    ``url`` : str
        URL from the file to be downloaded.
    ``fname`` : str
        Save name for the file.
    """
    
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

    """Script to automatically download the SSCS Dataset (converted to HDF5)
    if Dataset isn't found."""
    
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

    """Gets list of filenames from a subset of SSCS dataset.

    Parameters
    ----------
    ``split`` : str
        Subset to be retrieved (train, validate or test).
        Default is train.

    Returns
    -------
    ``split_list`` : list of str
        List of song filenames.
    """

    split_str = str(split).lower()
    if(split_str == 'train' or split_str == 'validate' or split_str == 'test'):
        split_list = json.load(open(splitname, 'r'))[split_str]
        return split_list
    else:
        raise NameError("Split should be 'train', 'validate' or 'test'.")
    
############################################################

def pick_songlist(first=0, amount=5, split='train'):

    """Picks a sequence of songs from a subset of SSCS Dataset.

    Parameters
    ----------
    ``first`` : int
        First song from the sequence to be picked. Default is 0.
    ``amount`` : int
        Number of songs in sequence to be picked. Default is 5.
    ``split`` : str
        Split from which the songs will be picked (train, validate or test).
        Default is train.

    Returns
    -------
    ``songnames`` : list of str
        List of song filenames.
    """
    
    songnames = get_split(split)
    return songnames

############################################################

def pick_random_song(split='train'):

    """Picks random song from a subset of SSCS Dataset.

    Parameters
    ----------
    ``split`` : str
        Split from which the song will be picked (train, validate or test).
        Default is train.

    Returns
    -------
    ``songname`` : str
        Random song filename.
    """
    
    songnames = get_split(split)
    rng = np.random.randint(0, len(songnames))
    return songnames[rng]

############################################################

def pick_multiple_random_songs(amount, split='train'):

    """Picks an amount of random songs from a subset of SSCS Dataset.

    Parameters
    ----------
    ``amount`` : int
        Amount of random songs to be picked.
    ``split`` : str
        Split from which the song will be picked (train, validate or test).
        Default is train.

    Returns
    -------
    ``songnames`` : list of str
        Random song filenames.
    """
    
    return [pick_random_song() for i in range(amount)]

############################################################

@ray.remote
def read_voice(name, voice):

    """Reads a voice from a song from SSCS Dataset.

    Parameters
    ----------
    ``name`` : str
        Name of the song.
    ``voice`` : str
        Voice to be read (mix, soprano, alto, tenor, bass).

    Returns
    -------
    ``timefreq_voice`` : pd.DataFrame
        Dataframe with resolution 360 x N with the time/frequency
        representation for the voice.
    """

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

    """Reads all voices from a song from SSCS Dataset.

    Parameters
    ----------
    ``name`` : str
        Name of the song.

    Returns
    -------
    ``mix`` : pd.DataFrame
        Dataframe with resolution 360 x N with the time/frequency
        representation for the mix voice.
    ``satb`` : list of pd.DataFrames
        List with four dataframes with resolution 360 x N containing
        time/frequency representations for soprano, alto, tenor and bass
        voices, in this order.
    """
    
    voices = ['mix', 'soprano', 'alto', 'tenor', 'bass']
    data_access = [read_voice.remote(name, voice) for voice in voices]
    df_voices = ray.get(data_access)
    mix = df_voices[0]
    satb = df_voices[1:]
    return mix, satb

############################################################

def split_and_reshape(df, split_size=SPLIT_SIZE):

    """Splits and reshapes a voice from a song

    Parameters
    ----------
    ``df`` : pd.DataFrame
        Dataframe with time/frequency representation of a voice
        from a song.
    ``split_size`` : int
        Size from each split, in time bins. Default is SPLIT_SIZE.

    Returns
    -------
    ``split_arr`` : ndarray
        Multidimensional array containing K splits of
        dimension 360 x ``split_size``. K depends on the
        duration of the song.
    """

    split_arr = np.array_split(df, df.shape[1]/split_size, axis=1)
    split_arr = np.array([i.iloc[:, :split_size] for i in split_arr])
    return split_arr

############################################################

def read_all_voice_splits(name, split_size=SPLIT_SIZE):

    """Reads all voices from a song and returns it splitted in 
    a fixed size of time bins per split.

    Parameters
    ----------
    ``name`` : str
        Name of the song to be retrieved.
    ``split_size`` : int
        Size from each split, in time bins. Default is SPLIT_SIZE.

    Returns
    -------
    ``mix_splits`` : ndarray
        Multidimensional array containing K splits of
        dimension 360 x ``split_size``.
    """

    mix_raw, satb_raw = read_all_voices(name)
    df_voices = satb_raw
    df_voices.insert(0, mix_raw)
    voice_splits = [split_and_reshape(df, split_size) for df in df_voices]
    mix_splits = voice_splits[0]
    s_splits = voice_splits[1]
    a_splits = voice_splits[2]
    t_splits = voice_splits[3]
    b_splits = voice_splits[4]
    return mix_splits, s_splits, a_splits, t_splits, b_splits

############################################################

def get_sequence(split='train', start_index=0, end_index=1000):
    """Returns a sequence with an amount of songs from a subset
    of SSCS Dataset. The sequence returns song splits with a
    fixed batch size and fixed time bin dimensions, so it can be
    used to compose a dataset ready for training.

    Parameters
    ----------
    ``split`` : str
        Subset from which the songs will be picked for the sequence
        (train, validate or test). Default is train.
    ``start_index`` : int 
        Index for the first song from the sequence (inclusive).
        Default is 0.
    ``end_index`` : int 
        Index for the last song from the sequence (exclusive).
        Default is 1000.

    Returns
    -------
    ``sequence`` : SSCS_Sequence
        Sequence of songs segmented in batches.
    """
    return SSCS_Sequence(get_split(split)[start_index:end_index])

############################################################

def get_dataset(split='train', start_index=0, end_index=1000):
    """Returns a dataset ready to use in training, based on
    a SSCS_Sequence.

    Parameters
    ----------
    ``split`` : str
        Subset from which the songs will be picked for the sequence
        (train, validate or test). Default is train.
    ``start_index`` : int 
        Index for the first song from the sequence (inclusive).
        Default is 0.
    ``end_index`` : int 
        Index for the last song from the sequence (exclusive).
        Default is 1000.

    Returns
    -------
    ``ds`` : tf.Dataset
        Tensorflow Dataset ready to use.
    """
    seq = get_sequence(split, 0, 2)
    mix_test, satb_test = seq.__getitem__(0)
    ds_spec = tf.TensorSpec(shape=mix_test.shape, dtype=TRAINING_DTYPE)
    signature = (ds_spec, (ds_spec, ds_spec, ds_spec, ds_spec))
    if(split=='train'):
        ds = tf.data.Dataset.from_generator(SSCS_Sequence,
                                    args = [get_split(split)[start_index:end_index]],
                                    output_signature=signature
                                    ).shuffle(10, seed=30,
                                              reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(SSCS_Sequence,
                                    args = [get_split(split)[start_index:end_index]],
                                    output_signature=signature
                                    ).prefetch(tf.data.AUTOTUNE)
    return ds

############################################################

def downsample_threshold(item):

    """Grabs a number representing the intensity of a frequency
    bin in a specific time bin, and binarizes it. 

    This function can be used when a frequency bins
    downsample is done.

    To use vectorized version, call
    ``vectorized_downsample_threshold``.

    Parameters
    ----------
    ``item`` : float
        Number representing the intensity of a frequency 

    Returns
    -------
    ``thresholded_item``: float
        1.0 if item >= 1.0; 0.0 otherwise
    """

    if item >= 1.0: return 1.0
    else: return 0.0

vectorized_downsample_threshold = np.vectorize(downsample_threshold)

############################################################

def downsample_limit(item):

    """Grabs a number representing the intensity of a frequency
    bin in a specific time bin, and limits it to the maximum
    value of 1.0. 

    This function can be used when a frequency bins
    downsample is done.

    To use vectorized version, call
    ``vectorized_downsample_limit``.

    Parameters
    ----------
    ``item`` : float
        Number representing the intensity of a frequency 

    Returns
    -------
    ``thresholded_item``: float
        1.0 if item >= 1.0; 0.0 otherwise
    """

    if item >= 1.0: return 1.0
    else: return item

vectorized_downsample_limit = np.vectorize(downsample_limit)

############################################################

def downsample_bins(voice):

    """Gets a time/frequency representation of a voice as input,
    and downsample it for 1/5 of frequency bin resolution.
    The upper and lower bins are discarded, because they're often
    filled with 1.0 value when the voice is in silence.

    Parameters
    ----------
    ``voice`` : ndarray
        360 x N matrix with time/frequency representation of a voice.

    Returns
    -------
    ``voice_sums``: ndarray
        69 x N matrix with downsampled time/frequency representation
        of a voice.
    """

    voice_0 = np.array(voice.T[0::5]).T
    voice_1 = np.array(voice.T[1::5]).T
    voice_2 = np.array(voice.T[2::5]).T
    voice_3 = np.array(voice.T[3::5]).T
    voice_4 = np.array(voice.T[4::5]).T

    voice_0 = voice_0.T[1:70].T
    voice_1 = voice_1.T[1:70].T
    voice_2 = voice_2.T[1:70].T
    voice_3 = voice_3.T[0:69].T
    voice_4 = voice_4.T[0:69].T

    voice_sums = voice_0 + voice_1 + voice_2 + voice_3 + voice_4
    voice_argmax = np.argmax(voice_sums, axis=1)
    threshold = np.zeros(voice_sums.shape)
    threshold[np.arange(voice_argmax.size), voice_argmax] = 1
    threshold[:, 0] = 0
    voice_sums = threshold

    #voice_sums = vectorized_downsample_threshold(voice_sums)

    return voice_sums

############################################################

def create_midi(freq, write_path='./MIDI/midi_track.mid', ticks_per_beat=58,
                tempo=90, save_to_file=True, program=53, channel=0):

    """Creates a single-channel MIDI file from a ndarray of frequencies in Hz.

    Parameters
    ----------
    ``freq`` : ndarray
        1-D array of frequencies in Hz.
    ``write_path`` : str
        Path where MIDI file will be saved. Default is ``./MIDI/midi_track.mid``.
    ``ticks_per_beat`` : int
        Time-length parameter for mido library. Default is 58.
    ``tempo`` : int
        BPM parameter for mido library. Default is 90.
    ``save_to_file`` : bool
        Choose whether to save a MIDI file or not. If True, the function
        saves a file in the path ``write_path`` and returns a MidiFile object.
        If False, the function only returns the MidiFile object.
        Default is True.
    ``program`` : int
        Number of the instrument (it can vary depending on the soundfont used
        to play the file). Default is 53.
    ``channel`` : int
        Instrument channel to record in the MIDI file. Default is 0.

    Returns
    -------
    ``midi``: MidiFile
        MidiFile object from mido library.
    """
    
    if(not os.path.exists(midi_dir)):
        os.mkdir(midi_dir)

    def freq_to_list(freq):
        # List event = (pitch, velocity, time)
        T = freq.shape[0]
        #midi_freqs = np.squeeze(midi_freqs)
        midi_freqs = np.round(69 + 12*np.log2(freq/440)).squeeze().astype('int')
        t_last = 0
        pitch_tm1 = 20
        list_event = []
        for t in range(T):
            pitch_t = midi_freqs[t]
            if (pitch_t != pitch_tm1):
                velocity = 127
                if(pitch_t == 24):
                    pitch_t = 0
                    velocity = 0
                t_event = t - t_last
                t_last = t
                list_event.append((pitch_tm1, 0, t_event))
                list_event.append((pitch_t, velocity, 0))
            pitch_tm1 = pitch_t
        list_event.append((pitch_tm1, 0, T - t_last))
        return list_event
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = mido.MidiFile()
    mid.ticks_per_beat = ticks_per_beat


    # Add a new track with the instrument name to the midi file
    track = mid.add_track("Voice Aah")
    # transform the matrix in a list of (pitch, velocity, time)
    events = freq_to_list(freq)
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

    """Creates a multi-channel MIDI file with all voices and saves it in
    the path ``write_path``.

    Parameters
    ----------
    ``sop`` : ndarray
        360 x N time/frequency representation for the soprano voice.
    ``alto`` : ndarray
        360 x N time/frequency representation for the alto voice.
    ``ten`` : ndarray
        360 x N time/frequency representation for the tenor voice.
    ``bass`` : ndarray
        360 x N time/frequency representation for the bass voice.
    ``write_path`` : str
        Path where MIDI file will be saved. Default is ``./MIDI/midi_mix.mid``.
    """

    bin_matrix = np.array([sop.T, alto.T, ten.T, bass.T])

    freq_matrix = bin_matrix_to_freq(bin_matrix)

    mid_sop = create_midi(freq_matrix[0], save_to_file=False, program=52, channel=0)
    mid_alto = create_midi(freq_matrix[1], save_to_file=False, program=53, channel=1)
    mid_ten = create_midi(freq_matrix[2], save_to_file=False, program=49, channel=2)
    mid_bass = create_midi(freq_matrix[3], save_to_file=False, program=50, channel=3)

    mid_mix = mido.MidiFile()
    mid_mix.ticks_per_beat=mid_sop.ticks_per_beat
    mid_mix.tracks = mid_sop.tracks + mid_alto.tracks + mid_ten.tracks + mid_bass.tracks
    mid_mix.save(write_path)

    return

############################################################

def songname_to_midi(songname, write_path=None):

    """Creates a multi-channel MIDI file from a song and saves it in
    the path ``write_path``.

    Parameters
    ----------
    ``songname`` : ndarray
        360 x N time/frequency representation for the bass voice.
    ``write_path`` : str
        Path where MIDI file will be saved. Default is ``./MIDI/`` + songname + ``.mid``.
    """

    if(not os.path.exists(midi_dir)):
        os.mkdir(midi_dir)

    mix, satb = read_all_voices(songname)
    sop = satb[0].to_numpy().T
    alto = satb[1].to_numpy().T
    ten = satb[2].to_numpy().T
    bass = satb[3].to_numpy().T
    mix = mix.T

    if(write_path is None):
        wrt = './MIDI/' + songname + '.mid'
        song_to_midi(sop, alto, ten, bass, write_path=wrt)
    else:
        song_to_midi(sop, alto, ten, bass, write_path=write_path)

    return

############################################################

def random_song_to_midi():

    """Picks a random song from SSCS Dataset and generates
    a MIDI file from this song."""

    song = pick_random_song()
    songname_to_midi(song)
    mix = ray.get(read_voice.remote(song, 'mix')).to_numpy()
    va_plots.plot(mix)
    return

############################################################

def load_weights(model, ckpt_dir=checkpoint_dir):

    """Loads weights for a specified model.

    Parameters
    ----------
    ``model`` : tf.Model
        Tensorflow model to load weights.
    ``ckpt_dir`` : str
        Path to file containing the weights to be loaded to ``model``.
    """

    if(os.path.exists(ckpt_dir)):
        model.load_weights(ckpt_dir)

############################################################

def train(model, ds_train, ds_val, epochs=EPOCHS,
          save_model=False, ckpt_dir=checkpoint_dir,
          log_folder='voas_cnn', early_stopping=None):
    
    """Trains a specified model.

    Parameters
    ----------
    ``model`` : tf.Model
        Tensorflow model to be trained.
    ``ds_train`` : tf.Dataset
        Dataset with train subset. This dataset can be get using the ``get_dataset`` function.
    ``ds_val`` : tf.Dataset
        Dataset with validation subset. This dataset can be get using the ``get_dataset`` function.
    ``epochs`` : int
        Number of epochs of the training. Default is EPOCHS.
    ``save_model`` : bool
        Choose to save (or not) the weights of the training. Default is False.
    ``ckpt_dir`` : str
        Path to weights file, which will be generated from the training process. Default is checkpoint_dir (check global vars).
    ``log_folder`` : str
        Folder name to save the training log files, which can be visualized on Tensorboard during and after the training.
    """
    
    logdir = log_dir + log_folder + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    callbacks = [tensorboard_cb]
    if(save_model):
        save_cb = tf.keras.callbacks.ModelCheckpoint(   filepath=ckpt_dir,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    monitor='val_loss',
                                                    save_best_only=True
                                                )
        callbacks.append(save_cb)
    if(early_stopping is not None):
        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=early_stopping)
        callbacks.append(early_cb)


    model.fit(  ds_train,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=ds_val)

############################################################

def prediction_postproc(input_array, argmax_and_threshold=True, gaussian_blur=True, high_pitch_fix=False):

    """Post-process the output of a model.

    Parameters
    ----------
    ``input_array`` : ndarray
        360 x N matrix containing time/frequency representation of a voice (output from a model prediction).
    ``argmax_and_threshold`` : bool
        Choose to apply (or not) argmax and threshold. Default is True.
    ``gaussian_blur`` : bool
        Choose to apply (or not) gaussian blur. This is only needed for visualization purposes. Default is True.
    ``high_pitch_fix`` : bool
        Choose to apply (or not) high pitch fix. The model may set the frequency bins 357~359 at a high value
        during a silence in the song, and this "fix" turns all the frequency bins above 357 to value 0.0.
        Default is False.

    Returns
    -------
    ``prediction`` : ndarray
        Post-processed time/frequency representation.
    """

    prediction = np.moveaxis(input_array, 0, 1).reshape(360, -1)
    if(argmax_and_threshold):
        prediction = np.argmax(prediction, axis=0)
        if(high_pitch_fix):
            prediction = np.array([i if i <= 357 else 0 for i in prediction])
        threshold = np.zeros((360, prediction.shape[0]))
        threshold[prediction, np.arange(prediction.size)] = 1
        prediction = threshold
    if(gaussian_blur):
        prediction = np.array(gaussian_filter1d(prediction, 1, axis=0, mode='wrap'))
        prediction = (prediction - np.min(prediction))/(np.max(prediction)-np.min(prediction))
    return prediction

############################################################

freqscale = librosa.cqt_frequencies(n_bins=360, fmin=32.7, bins_per_octave=60)

def bin_to_freq(bin):
    """Converts a bin value to a frequency value in Hz.

    To use the vectorized version, call ``vec_bin_to_freq``.

    Parameters
    ----------
    ``bin`` : int
        The frequency bin number (between 0 and 359).

    Returns
    -------
    ``freq`` : float
        Frequency value in Hz.
    """
    return freqscale[bin]

vec_bin_to_freq = np.vectorize(bin_to_freq)

############################################################

def resample_timescale(freqs, ref_timescale):

    """Resample timescale to common timebase (needed for metrics evaluation).
    This function can be useful when calculating metrics from a frequency list
    with a different timebase than the used on SSCS Dataset.

    Parameters
    ----------
    ``freqs`` : ndarray
        Array of frequencies to be resampled.
    ``ref_timescale`` : ndarray
        Array with timestamps for each frequency in ``freqs``.

    Returns
    -------
    ``output_freqs`` : ndarray
        Resampled freqs.
    """

    max_time = ref_timescale[-1]
    timescale = np.arange(0, max_time, 0.011609977)
    freqs_reshape = [np.array([i]) for i in freqs.reshape(-1)]
    output_freqs = np.array([mir_eval.multipitch.resample_multipitch(ref_timescale, freqs_reshape, timescale)]).reshape(-1, 1)
    return output_freqs

############################################################

def bin_matrix_to_freq(matrix, ref_timescale=None):

    """Converts a matrix containing frequency bin values over time
    into a matrix containing frequency values in Hz over time.
    The matrix should have a shape (4, N), with N depending on
    the duration of the song.

    matrix[0], matrix[1], matrix[2], matrix[3] represents the
    soprano, alto, tenor and bass voices, respectively.

    Parameters
    ----------
    ``matrix`` : ndarray
        Matrix containing frequency bin values for each time step.
    ``ref_timescale`` : ndarray
        Array with timestamps for each frequency in ``matrix``. If ``ref_timescale``
        is set, the matrix is resampled. Default is ``None``.

    Returns
    -------
    ``freqs`` : ndarray
        Matrix converted to frequency values in Hz.
    """

    s_freqs_raw = vec_bin_to_freq(np.argmax(matrix[0], axis=0)).reshape(-1, 1)
    a_freqs_raw = vec_bin_to_freq(np.argmax(matrix[1], axis=0)).reshape(-1, 1)
    t_freqs_raw = vec_bin_to_freq(np.argmax(matrix[2], axis=0)).reshape(-1, 1)
    b_freqs_raw = vec_bin_to_freq(np.argmax(matrix[3], axis=0)).reshape(-1, 1)

    if(ref_timescale is None):
        s_freqs = np.copy(s_freqs_raw)
        a_freqs = np.copy(a_freqs_raw)
        t_freqs = np.copy(t_freqs_raw)
        b_freqs = np.copy(b_freqs_raw)
    else:
        s_freqs = resample_timescale(s_freqs_raw, ref_timescale)
        a_freqs = resample_timescale(a_freqs_raw, ref_timescale)
        t_freqs = resample_timescale(t_freqs_raw, ref_timescale)
        b_freqs = resample_timescale(b_freqs_raw, ref_timescale)
        
    freqs = np.concatenate((s_freqs, a_freqs, t_freqs, b_freqs), axis=1).T
    return freqs

############################################################

def f_score(precision, recall):

    """Calculates F-Score metric from Precision and Recall values.

    Parameters
    ----------
    ``precision`` : float
        Precision metric value.
    ``recall`` : float
        Recall metric value.

    Returns
    -------
    ``fscore`` : float
        F-Score metric value.
    """

    return 2 * (precision * recall) / (precision + recall + K.epsilon())

############################################################

def __metrics_aux(ref_time, ref_freqs, est_time, est_freqs):

    """Auxiliar function to compute evaluation metrics for the model.

    This function gets as input the frequencies from the reference annotation
    of a voice from a song and the estimated frequency annotation of this song
    from a model along with them respective timestamp arrays, and retrieve a
    dataframe with metrics calculated with the ``mir_eval`` library.

    Parameters
    ----------
    ``ref_time`` : ndarray
        Array with reference annotation timestamp.
    ``ref_freqs`` : ndarray
        Array with reference annotation frequencies.
    ``est_time`` : ndarray
        Array with estimated annotation timestamp.
    ``est_freqs`` : ndarray
        Array with estimated annotation frequencies.

    Returns
    -------
    ``metrics_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated with ``mir_eval`` library.
    """

    multipitch_metrics = mir_eval.multipitch.evaluate(ref_time, ref_freqs, est_time, est_freqs)
    melody_metrics = mir_eval.melody.evaluate(ref_time, np.squeeze(ref_freqs), est_time, np.squeeze(est_freqs))
    multipitch_metrics['F-Measure'] = f_score(multipitch_metrics['Precision'], multipitch_metrics['Recall'])
    metrics_dict = multipitch_metrics
    metrics_dict.update(melody_metrics)
    metrics_df = pd.DataFrame([metrics_dict]).astype('float64')
    return metrics_df

############################################################

def metrics(y_true_matrix, y_pred_matrix, true_timescale=None):

    """Function to calculate evaluation metrics for all the voices of a song.

    This function gets as input the matrix with all the reference voice annotations
    and a matrix with all the estimated voice annotations. If needed, the timescale
    for the reference annotations can be passed as an argument, and the reference
    annotations will be resampled.
    
    The function returns five dataframes with the metrics calculated for the mix
    and for the individual voices. All metrics are calculated using the
    ``mir_eval`` library.

    Parameters
    ----------
    ``y_true_matrix`` : ndarray
        Matrix containing the reference frequency annotations for each voice
        (soprano, alto, tenor and bass, in this order).
    ``y_pred_matrix`` : ndarray
        Matrix containing the estimated frequency annotations for each voice
        (soprano, alto, tenor and bass, in this order).
    ``true_timescale`` : ndarray
        If set, the reference frequency annotations will be resampled.
        Default is ``None``.

    Returns
    -------
    ``mix_metrics_df`` : pd.DataFrame
        Dataframe with multipitch metrics calculated for mixed voices with
        ``mir_eval`` library.
    ``s_metrics_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Soprano voice
        with ``mir_eval`` library.
    ``a_metrics_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Alto voice
        with ``mir_eval`` library.
    ``t_metrics_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Tenor voice
        with ``mir_eval`` library.
    ``b_metrics_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Bass voice
        with ``mir_eval`` library.
    """

    timescale = np.arange(0, 0.011609977 * (y_pred_matrix.shape[1]), 0.011609977)[:y_pred_matrix.shape[1]]
    
    if(true_timescale is None):
        s_true_freqs = y_true_matrix[0].reshape(-1, 1)
        a_true_freqs = y_true_matrix[1].reshape(-1, 1)
        t_true_freqs = y_true_matrix[2].reshape(-1, 1)
        b_true_freqs = y_true_matrix[3].reshape(-1, 1)
    else:
        s_true_freqs = resample_timescale(y_true_matrix[0].reshape(-1, 1), true_timescale)[:y_pred_matrix.shape[1]]
        a_true_freqs = resample_timescale(y_true_matrix[1].reshape(-1, 1), true_timescale)[:y_pred_matrix.shape[1]]
        t_true_freqs = resample_timescale(y_true_matrix[2].reshape(-1, 1), true_timescale)[:y_pred_matrix.shape[1]]
        b_true_freqs = resample_timescale(y_true_matrix[3].reshape(-1, 1), true_timescale)[:y_pred_matrix.shape[1]]
       
    y_true_freqs = np.concatenate((s_true_freqs, a_true_freqs, t_true_freqs, b_true_freqs), axis=1)

    s_pred_freqs = y_pred_matrix[0].reshape(-1, 1)
    a_pred_freqs = y_pred_matrix[1].reshape(-1, 1)
    t_pred_freqs = y_pred_matrix[2].reshape(-1, 1)
    b_pred_freqs = y_pred_matrix[3].reshape(-1, 1)

    #y_pred_freqs = np.concatenate((s_pred_freqs, a_pred_freqs, t_pred_freqs, b_pred_freqs), axis=1)

    s_metrics_df = __metrics_aux(timescale, s_true_freqs, timescale, s_pred_freqs)
    a_metrics_df = __metrics_aux(timescale, a_true_freqs, timescale, a_pred_freqs)
    t_metrics_df = __metrics_aux(timescale, t_true_freqs, timescale, t_pred_freqs)
    b_metrics_df = __metrics_aux(timescale, b_true_freqs, timescale, b_pred_freqs)
    
    mix_multipitch_metrics = mir_eval.multipitch.evaluate(timescale, y_true_freqs, timescale, y_pred_matrix.T)
    mix_multipitch_metrics['F-Measure'] = f_score(mix_multipitch_metrics['Precision'], mix_multipitch_metrics['Recall'])
    mix_metrics_df = pd.DataFrame([mix_multipitch_metrics]).astype('float64')

    return mix_metrics_df, s_metrics_df, a_metrics_df, t_metrics_df, b_metrics_df

############################################################

def metrics_test_precompute(model, save_dir):

    """Precompute metrics for a choosen model for each song in the test subset
    of the SSCS Dataset. Saves the precomputed metrics as a HDF5 file in a 
    given file path.

    Parameters
    ----------
    ``model`` : tf.Model
        Model with trained weights to be evaluated.
    ``save_dir`` : str
        Path to file in which the precomputed metrics will be saved.

    Returns
    -------
    ``mix_df`` : pd.DataFrame
        Dataframe with multipitch metrics calculated for mixed voices with
        ``mir_eval`` library.
    ``sop_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Soprano voice
        with ``mir_eval`` library.
    ``alto_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Alto voice
        with ``mir_eval`` library.
    ``ten_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Tenor voice
        with ``mir_eval`` library.
    ``bass_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Bass voice
        with ``mir_eval`` library.
    """

    #set amount to 805 to calculate metrics to entire test set
    amount = 805
    songs = pick_songlist(amount=amount, split='test')

    true_f0 = []
    pred_f0 = []

    mix_df = pd.DataFrame()
    sop_df = pd.DataFrame()
    alto_df = pd.DataFrame()
    ten_df = pd.DataFrame()
    bass_df = pd.DataFrame()

    counter = 1

    for song in songs:
        print(f"Predicting on Model {model.name}, Song {counter}/{amount}")
        counter += 1
        voice_splits = read_all_voice_splits(song)
        voice_pred = model.predict(voice_splits[0])
        true_grids = np.array([np.moveaxis(split, 0, 1).reshape(360, -1) for split in voice_splits])[1:]
        pred_grids = np.array([prediction_postproc(pred).astype(np.float32) for pred in voice_pred])
        true_f0.append(bin_matrix_to_freq(true_grids))
        pred_f0.append(bin_matrix_to_freq(pred_grids))

    print(f"Calculating metrics for Model {model.name}...")

    @ray.remote
    def precompute_calc(true_freq, pred_freq, songname):
        mix_song_df, s_song_df, a_song_df, t_song_df, b_song_df = metrics(true_freq, pred_freq)
        mix_song_df.insert(loc=0, column='Songname', value=songname)
        s_song_df.insert(loc=0, column='Songname', value=songname)
        a_song_df.insert(loc=0, column='Songname', value=songname)
        t_song_df.insert(loc=0, column='Songname', value=songname)
        b_song_df.insert(loc=0, column='Songname', value=songname)
        return [mix_song_df, s_song_df, a_song_df, t_song_df, b_song_df]

    precompute_array = [precompute_calc.remote(ray.put(true_f0[idx]),
                                               ray.put(pred_f0[idx]),
                                               ray.put(songs[idx])) for idx in range(amount)]    
    precompute = ray.get(precompute_array)

    mix_df = pd.concat([precompute[i][0] for i in range(amount)], axis=0, ignore_index=True)
    sop_df = pd.concat([precompute[i][1] for i in range(amount)], axis=0, ignore_index=True)
    alto_df = pd.concat([precompute[i][2] for i in range(amount)], axis=0, ignore_index=True)
    ten_df = pd.concat([precompute[i][3] for i in range(amount)], axis=0, ignore_index=True)
    bass_df = pd.concat([precompute[i][4] for i in range(amount)], axis=0, ignore_index=True)

    mix_df.to_hdf(save_dir, 'mix', mode='w', complevel=9, complib='blosc', append=False, format='table')
    sop_df.to_hdf(save_dir, 'soprano', mode='a', complevel=9, complib='blosc', append=True, format='table')
    alto_df.to_hdf(save_dir, 'alto', mode='a', complevel=9, complib='blosc', append=True, format='table')
    ten_df.to_hdf(save_dir, 'tenor', mode='a', complevel=9, complib='blosc', append=True, format='table')
    bass_df.to_hdf(save_dir, 'bass', mode='a', complevel=9, complib='blosc', append=True, format='table')
    
    return mix_df, sop_df, alto_df, ten_df, bass_df

############################################################

def metrics_load_precomputed(file_path):

    """Load precomputed metrics from a HDF5 file in a given file path.

    Parameters
    ----------
    ``file_path`` : str
        Path to file from which the precomputed metrics will be loaded.

    Returns
    -------
    ``mix_df`` : pd.DataFrame
        Dataframe with multipitch metrics calculated for mixed voices with
        ``mir_eval`` library.
    ``sop_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Soprano voice
        with ``mir_eval`` library.
    ``alto_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Alto voice
        with ``mir_eval`` library.
    ``ten_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Tenor voice
        with ``mir_eval`` library.
    ``bass_df`` : pd.DataFrame
        Dataframe with melody and multipitch metrics calculated for Bass voice
        with ``mir_eval`` library.
    """

    mix_df = pd.read_hdf(file_path, key='mix', mode='r')
    sop_df = pd.read_hdf(file_path, key='soprano', mode='r')
    alto_df = pd.read_hdf(file_path, key='alto', mode='r')
    ten_df = pd.read_hdf(file_path, key='tenor', mode='r')
    bass_df = pd.read_hdf(file_path, key='bass', mode='r')
    return mix_df, sop_df, alto_df, ten_df, bass_df

############################################################

def playground(model):
    """Arbitrary function only for demonstration and "play around and find out" purposes.

    Parameters
    ----------
    ``model`` : tf.Model
        Pretrained model for play around.
    """
    rand_song = pick_random_song(split='test')
    mix, s, a, t, b = read_all_voice_splits(rand_song)

    s_pred, a_pred, t_pred, b_pred = model.predict(mix)

    mix = np.moveaxis(mix, 0, 1).reshape(360, -1)
    s = np.moveaxis(s, 0, 1).reshape(360, -1)
    a = np.moveaxis(a, 0, 1).reshape(360, -1)
    t = np.moveaxis(t, 0, 1).reshape(360, -1)
    b = np.moveaxis(b, 0, 1).reshape(360, -1)

    y_true = np.array([s, a, t, b])

    s_pred_postproc = prediction_postproc(s_pred).astype(np.float32)
    a_pred_postproc = prediction_postproc(a_pred).astype(np.float32)
    t_pred_postproc = prediction_postproc(t_pred).astype(np.float32)
    b_pred_postproc = prediction_postproc(b_pred).astype(np.float32)
    mix_pred_postproc = s_pred_postproc + a_pred_postproc + t_pred_postproc + b_pred_postproc
    mix_pred_postproc = vectorized_downsample_limit(mix_pred_postproc)

    y_pred_postproc = np.array([[s_pred_postproc], [a_pred_postproc], [t_pred_postproc], [b_pred_postproc]])
    y_pred_postproc = np.squeeze(y_pred_postproc)

    y_true_freqs = bin_matrix_to_freq(y_true)
    y_pred_freqs = bin_matrix_to_freq(y_pred_postproc)
    
    song_metrics = metrics(y_true_freqs, y_pred_freqs)

    mix_fscore = song_metrics[0]['F-Measure'].to_numpy()[0]
    s_fscore = song_metrics[1]['F-Measure'].to_numpy()[0]
    a_fscore = song_metrics[2]['F-Measure'].to_numpy()[0]
    t_fscore = song_metrics[3]['F-Measure'].to_numpy()[0]
    b_fscore = song_metrics[4]['F-Measure'].to_numpy()[0]

    print(f"Song: {rand_song}")
    print("===================")
    print("F-Scores:")
    print(f"Soprano: {s_fscore}")
    print(f"Alto: {a_fscore}")
    print(f"Tenor: {t_fscore}")
    print(f"Bass: {b_fscore}")
    print(f"Mix: {mix_fscore}")
    print()

    va_plots.plot(mix, title='Mix - Ground Truth')
    va_plots.plot(mix_pred_postproc, title='Mix - Rebuilt from predictions from ' + model.name)

    va_plots.plot(s, title='Soprano - Ground Truth')
    va_plots.plot(s_pred_postproc, title='Soprano - Prediction from ' + model.name)

    va_plots.plot(a, title='Alto - Ground Truth')
    va_plots.plot(a_pred_postproc, title='Alto - Prediction from ' + model.name)

    va_plots.plot(t, title='Tenor - Ground Truth')
    va_plots.plot(t_pred_postproc, title='Tenor - Prediction from ' + model.name)

    va_plots.plot(b, title='Bass - Ground Truth')
    va_plots.plot(b_pred_postproc, title='Bass - Prediction from ' + model.name)

    songname_to_midi(rand_song, write_path='./MIDI/original.mid')
    song_to_midi(s_pred_postproc.T, a_pred_postproc.T, t_pred_postproc.T, b_pred_postproc.T, './MIDI/predicted.mid')

############################################################
