import os
import math
import mido
import pumpp
import spaces
import librosa
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from cqfe_models import mask_voas_cnn_v2_model, late_deep_cnn_model

SATB_THRESHOLDS = [0.23, 0.17, 0.15, 0.17]

############################################################

freqscale = librosa.cqt_frequencies(n_bins=360, fmin=32.7, bins_per_octave=60)

def bin_to_freq(bin):
    return freqscale[bin]

vec_bin_to_freq = np.vectorize(bin_to_freq)

############################################################

def downsample_bins(voice):
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

    return voice_sums

############################################################

def bin_matrix_to_freq(matrix):
    s_freqs = vec_bin_to_freq(np.argmax(matrix[0], axis=0)).reshape(-1, 1)
    a_freqs = vec_bin_to_freq(np.argmax(matrix[1], axis=0)).reshape(-1, 1)
    t_freqs = vec_bin_to_freq(np.argmax(matrix[2], axis=0)).reshape(-1, 1)
    b_freqs = vec_bin_to_freq(np.argmax(matrix[3], axis=0)).reshape(-1, 1)
        
    freqs = np.concatenate((s_freqs, a_freqs, t_freqs, b_freqs), axis=1).T
    return freqs

############################################################

def create_midi(freq, write_path='./midi_track.mid', ticks_per_beat=58,
                tempo=90, save_to_file=True, program=53, channel=0):
    
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

def song_to_midi(sop, alto, ten, bass):

    savepath = './output.mid'

    bin_matrix = np.array([sop, alto, ten, bass])
    freq_matrix = bin_matrix_to_freq(bin_matrix)

    mid_sop = create_midi(freq_matrix[0], save_to_file=False, program=52, channel=0)
    mid_alto = create_midi(freq_matrix[1], save_to_file=False, program=52, channel=1)
    mid_ten = create_midi(freq_matrix[2], save_to_file=False, program=52, channel=2)
    mid_bass = create_midi(freq_matrix[3], save_to_file=False, program=52, channel=3)

    mid_mix = mido.MidiFile()
    mid_mix.ticks_per_beat=mid_sop.ticks_per_beat
    mid_mix.tracks = mid_sop.tracks + mid_alto.tracks + mid_ten.tracks + mid_bass.tracks
    mid_mix.save(savepath)

    return savepath

############################################################

def song_to_dataframe(sop, alto, ten, bass):

    timescale = np.arange(0, 0.011609977 * (sop.shape[1]), 0.011609977)[:sop.shape[1]]

    s_argmax = vec_bin_to_freq(np.argmax(sop, axis=0))
    a_argmax = vec_bin_to_freq(np.argmax(alto, axis=0))
    t_argmax = vec_bin_to_freq(np.argmax(ten, axis=0))
    b_argmax = vec_bin_to_freq(np.argmax(bass, axis=0))

    data = np.array([timescale, s_argmax, a_argmax, t_argmax, b_argmax], dtype=np.float32).T
    columns = ['Timestep', 'Soprano', 'Alto', 'Tenor', 'Bass']

    df = pd.DataFrame(data, columns=columns)

    return df

############################################################

def prediction_postproc(input_array, argmax_and_threshold=True,
                                     gaussian_blur=True,
                                     threshold_value=0):
    prediction = np.moveaxis(input_array, 0, 1).reshape(360, -1)
    thres_reference = deepcopy(prediction)
    if(argmax_and_threshold):
        prediction = np.argmax(prediction, axis=0)
        prediction = np.array([prediction[i] if thres_reference[prediction[i], i] >= threshold_value else 0 for i in np.arange(prediction.size)])
        threshold = np.zeros((360, prediction.shape[0]))
        threshold[prediction, np.arange(prediction.size)] = 1
        prediction = threshold
    if(gaussian_blur):
        prediction = np.array(gaussian_filter1d(prediction, 1, axis=0, mode='wrap'))
        prediction = (prediction - np.min(prediction))/(np.max(prediction)-np.min(prediction))
    return prediction

############################################################

def get_hcqt_params():

    bins_per_octave = 60
    n_octaves = 6
    over_sample = 5
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256

    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample

############################################################

def create_pump_object():

    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length, over_sample) = get_hcqt_params()

    p_phdif = pumpp.feature.HCQTPhaseDiff(name='dphase', sr=sr, hop_length=hop_length,
                                   fmin=f_min, n_octaves=n_octaves, over_sample=over_sample, harmonics=harmonics, log=True)

    pump = pumpp.Pump(p_phdif)

    return pump

############################################################

def compute_pump_features(pump, audio_fpath):

    data = pump(audio_f=audio_fpath)

    return data

############################################################

@spaces.GPU(duration=600)
def get_mpe_prediction(model, audio_file=None):
    """Generate output from a model given an input numpy file.
       Part of this function is part of deepsalience
    """

    split_value = 4000

    if audio_file is not None:

        pump = create_pump_object()
        features = compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

    else:
        raise ValueError("One audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[3]
    t_slices = list(np.arange(0, n_t, split_value))
    output_list = []

    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, :, t:t+split_value], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, :, t:t+split_value], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list).astype(np.float32)

    return predicted_output

############################################################

@spaces.GPU(duration=600)
def get_va_prediction(model, f0_matrix):
    splits = f0_matrix.shape[1]//256
    splits_diff = 256 - (f0_matrix.shape[1] - splits * 256)
    fill = np.zeros((360, splits_diff))
    mix_filled = np.concatenate((np.copy(f0_matrix), fill), axis=1)
    mix_filled = np.reshape(mix_filled, (360, -1, 256, 1)).transpose((1, 0, 2, 3))
    batches = math.ceil(mix_filled.shape[0]/24)

    s_pred_result = np.zeros((0, 360, 256))
    a_pred_result = np.zeros((0, 360, 256))
    t_pred_result = np.zeros((0, 360, 256))
    b_pred_result = np.zeros((0, 360, 256))

    for i in range(batches):
        s_pred, a_pred, t_pred, b_pred = model.predict(mix_filled[i*24:(i+1)*24])
        s_pred_result = np.append(s_pred_result, s_pred, axis=0)
        a_pred_result = np.append(a_pred_result, a_pred, axis=0)
        t_pred_result = np.append(t_pred_result, t_pred, axis=0)
        b_pred_result = np.append(b_pred_result, b_pred, axis=0)

    s_pred_result = prediction_postproc(s_pred_result, threshold_value=SATB_THRESHOLDS[0])[:, :f0_matrix.shape[1]]
    a_pred_result = prediction_postproc(a_pred_result, threshold_value=SATB_THRESHOLDS[1])[:, :f0_matrix.shape[1]]
    t_pred_result = prediction_postproc(t_pred_result, threshold_value=SATB_THRESHOLDS[2])[:, :f0_matrix.shape[1]]
    b_pred_result = prediction_postproc(b_pred_result, threshold_value=SATB_THRESHOLDS[3])[:, :f0_matrix.shape[1]]

    return s_pred_result, a_pred_result, t_pred_result, b_pred_result

############################################################

def cqfe(audiofile, mpe=late_deep_cnn_model(), va=mask_voas_cnn_v2_model()):
    
    savepath_csv = './output.csv'
    savepath_hdf5 = './output.hdf5'

    mpe_pred = get_mpe_prediction(mpe, audiofile)
    s_pred, a_pred, t_pred, b_pred = get_va_prediction(va, mpe_pred)

    output_midi = song_to_midi(s_pred, a_pred, t_pred, b_pred)

    output_df = song_to_dataframe(s_pred, a_pred, t_pred, b_pred)
    output_df.to_csv(savepath_csv, mode='w', header=True)
    output_df.to_hdf(savepath_hdf5, key='F0', mode='w', complevel=9, complib='blosc', append=False, format='table')
    ax1 = output_df.plot.scatter(x='Timestep', y='Bass', s=1, color='#2f29e3', label='Bass')
    ax2 = output_df.plot.scatter(x='Timestep', y='Tenor', s=1, color='#e36129', label='Tenor', ax=ax1)
    ax3 = output_df.plot.scatter(x='Timestep', y='Alto', s=1, color='#29e35a', label='Alto', ax=ax1)
    ax4 = output_df.plot.scatter(x='Timestep', y='Soprano', s=1, color='#d3d921', label='Soprano', ax=ax1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Freq (Hz)')
    fig = ax1.get_figure()
    fig.set_dpi(150)

    return [output_midi, savepath_csv, savepath_hdf5], fig

############################################################
