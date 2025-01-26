"""
File modified by André Paiva
"""

from __future__ import print_function
import models
import utils
import utils_train

import hdf5plugin
import numpy as np
import pandas as pd
import scipy

import os
import argparse

def get_single_test_prediction(model, audio_file=None):
    """Generate output from a model given an input numpy file.
       Part of this function is part of deepsalience
    """

    if audio_file is not None:

        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

    else:
        raise ValueError("One audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[3]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []

    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, :, t:t+5000], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, :, t:t+5000], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list).astype(np.float32)
    return predicted_output


def main(args):

    audiofile = args.audiofile
    audio_folder = args.audio_folder

    save_key = 'exp3multif0'
    model_path = "./models/{}.h5".format(save_key)
    model = models.build_model3()
    model.load_weights(model_path)
    thresh = 0.5
    peak_picking = False

    # compile model

    model.compile(
        loss=utils_train.bkld, metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )
    print("Model compiled")

    # select operation mode and compute prediction
    if audiofile != "0":

        # predict using trained model
        predicted_output = get_single_test_prediction(
            model, audio_file=audiofile
        )

        if(peak_picking):
            peak_thresh_mat = np.zeros(predicted_output.shape)
            peaks = scipy.signal.argrelmax(predicted_output, axis=0)
            peak_thresh_mat[peaks] = predicted_output[peaks] >= thresh
            predicted_output = peak_thresh_mat.astype(np.float32)

        df = pd.DataFrame(predicted_output.T)
        df.to_hdf(audiofile.replace('wav', 'h5'), 'mix', mode='a', complevel=9, complib='blosc', append=False, format='table')
        
        print(" > > > Multiple F0 prediction for {} exported as {}.".format(
            audiofile, audiofile.replace('wav', 'h5'))
        )

    elif audio_folder != "0":

        for audiofile in os.listdir(audio_folder):

            if not audiofile.endswith('wav'): continue

            # predict using trained model
            predicted_output = get_single_test_prediction(
                 model, audio_file=os.path.join(
                    audio_folder, audiofile)
            )

            if(peak_picking):
                peak_thresh_mat = np.zeros(predicted_output.shape)
                peaks = scipy.signal.argrelmax(predicted_output, axis=0)
                peak_thresh_mat[peaks] = predicted_output[peaks] >= thresh
                predicted_output = peak_thresh_mat.astype(np.float32)

            df = pd.DataFrame(predicted_output.T)
            df.to_hdf(os.path.join(audio_folder, audiofile.replace('wav', 'h5')), 'mix', mode='a', complevel=9, complib='blosc', append=False, format='table')

            print(" > > > Multiple F0 prediction for {} exported as {}.".format(
                audiofile, os.path.join(
                    audio_folder, audiofile.replace('wav', 'h5')
                ))
            )
    else:
        raise ValueError("One of audiofile and audio_folder must be specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of an input audio file or all the audio files inside a folder.")

    parser.add_argument("--audiofile",
                        dest='audiofile',
                        default="0",
                        type=str,
                        help="Path to the audio file to analyze. If using the folder mode, this should be skipped.")

    parser.add_argument("--audio_folder",
                        dest='audio_folder',
                        default="0",
                        type=str,
                        help="Directory with audio files to analyze. If using the audiofile mode, this should be skipped.")

    main(parser.parse_args())
