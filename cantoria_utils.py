import math
import numpy as np
import pandas as pd
import va_utils
import librosa
import mir_eval

TRUE_THRES = 20
PRED_THRES = 32.7

############################################################

def voice_metrics(y_true, y_pred,
                  timescale):
    """Calculates multipitch and melody metrics with ``mir_eval`` library.

    Parameters
    ----------
    ``y_true`` : np.ndarray
        Array of reference frequencies
    ``y_pred`` : np.ndarray
        Array of predicted frequencies
    ``timescale`` : np.ndarray
        Array of time stamps

    Returns
    -------
    ``metrics_df`` : pd.DataFrame
        Dataframe containing calculated metrics
    """
    y_true_with_silence = [np.array([]) if freq < TRUE_THRES else np.array([freq]) for freq in y_true]
    y_pred_with_silence = [np.array([]) if freq <= PRED_THRES else np.array([freq]) for freq in y_pred]
    multipitch_metrics = mir_eval.multipitch.evaluate(timescale, y_true_with_silence,
                                                      timescale, y_pred_with_silence)
    multipitch_metrics['F-Measure'] = va_utils.f_score(multipitch_metrics['Precision'],
                                                       multipitch_metrics['Recall'])
    melody_metrics =  mir_eval.melody.evaluate(timescale, np.array(y_true).squeeze(),
                                               timescale, np.array(y_pred).squeeze())
    metrics_dict = multipitch_metrics
    metrics_dict.update(melody_metrics)
    metrics_df = pd.DataFrame([metrics_dict]).astype('float64')
    return metrics_df

############################################################

def mix_metrics(s_true, s_pred,
                a_true, a_pred,
                t_true, t_pred,
                b_true, b_pred,
                timescale):
    
    """Calculates multipitch metrics with ``mir_eval`` library.

    Parameters
    ----------
    ``s_true`` : np.ndarray
        Array of reference frequencies for soprano
    ``s_pred`` : np.ndarray
        Array of predicted frequencies for soprano
    ``a_true`` : np.ndarray
        Array of reference frequencies for alto
    ``a_pred`` : np.ndarray
        Array of predicted frequencies for alto
    ``t_true`` : np.ndarray
        Array of reference frequencies for tenor
    ``t_pred`` : np.ndarray
        Array of predicted frequencies for tenor
    ``b_true`` : np.ndarray
        Array of reference frequencies for bass
    ``b_pred`` : np.ndarray
        Array of predicted frequencies for bass
    ``timescale`` : np.ndarray
        Array of time stamps

    Returns
    -------
    ``metrics_df`` : pd.DataFrame
        Dataframe containing calculated metrics
    """
    
    s_true_with_silence = [np.array([]) if freq < TRUE_THRES else np.array([freq]) for freq in s_true]
    a_true_with_silence = [np.array([]) if freq < TRUE_THRES else np.array([freq]) for freq in a_true]
    t_true_with_silence = [np.array([]) if freq < TRUE_THRES else np.array([freq]) for freq in t_true]
    b_true_with_silence = [np.array([]) if freq < TRUE_THRES else np.array([freq]) for freq in b_true]
    mix_true = [np.concatenate((s_true_with_silence[i],
                                a_true_with_silence[i],
                                t_true_with_silence[i],
                                b_true_with_silence[i])) for i in range(len(s_true))]

    s_pred_with_silence = [np.array([]) if freq <= PRED_THRES else np.array([freq]) for freq in s_pred]
    a_pred_with_silence = [np.array([]) if freq <= PRED_THRES else np.array([freq]) for freq in a_pred]
    t_pred_with_silence = [np.array([]) if freq <= PRED_THRES else np.array([freq]) for freq in t_pred]
    b_pred_with_silence = [np.array([]) if freq <= PRED_THRES else np.array([freq]) for freq in b_pred]
    mix_pred = [np.concatenate((s_pred_with_silence[i],
                                a_pred_with_silence[i],
                                t_pred_with_silence[i],
                                b_pred_with_silence[i])) for i in range(len(s_pred))]
    
    multipitch_metrics = mir_eval.multipitch.evaluate(timescale, mix_true, timescale, mix_pred)
    multipitch_metrics['F-Measure'] = va_utils.f_score(multipitch_metrics['Precision'],
                                                       multipitch_metrics['Recall'])
    metrics_df = pd.DataFrame([multipitch_metrics]).astype('float64')
    return metrics_df

############################################################

def cantoria_metrics(model_type: int):

    """Calculates metrics for Cantoría Dataset predicted multi-pitch estimations
    with ``mir_eval`` library. The metrics are computed for each voice and for
    the mix. 

    Parameters
    ----------
    ``model_type`` : int
        Number that represents the model.
        0: MaskVoasCNN
        1: MaskVoasCNNv2
        2: DownsampleVoasCNN
        3: DownsampleVoasCNNv2
        4: VoasCNN (Retrained weights)
        5: VoasCNN (Original weights)

    Returns
    -------
    ``mix_df`` : pd.DataFrame
        Dataframe containing calculated metrics for mix
    ``sop_df`` : pd.DataFrame
        Dataframe containing calculated metrics for soprano
    ``alto_df`` : pd.DataFrame
        Dataframe containing calculated metrics for alto
    ``ten_df`` : pd.DataFrame
        Dataframe containing calculated metrics for tenor
    ``bass_df`` : pd.DataFrame
        Dataframe containing calculated metrics for bass
    """

    songlist = ['CEA', 'EJB1', 'EJB2', 'HCB', 'LBM1', 'LBM2', 'LJT1', 'LJT2', 'LNG', 'RRC', 'SSS', 'THM', 'VBP', 'YSM']
    if (model_type == 0):
        ckpt_dir = './Checkpoints/mask_voas_treino1.keras'
        model = va_utils.mask_voas_cnn_model()
    elif (model_type == 1):
        ckpt_dir = './Checkpoints/mask_voas_v2_treino1.keras'
        model = va_utils.mask_voas_cnn_v2_model()
    elif (model_type == 2):
        ckpt_dir = './Checkpoints/downsample_voas_treino1.keras'
        model = va_utils.downsample_voas_cnn_model()
    elif (model_type == 3):
        ckpt_dir = './Checkpoints/downsample_voas_v2_treino1.keras'
        model = va_utils.downsample_voas_cnn_v2_model()
    elif (model_type == 4):
        ckpt_dir = './Checkpoints/voas_treino1.keras'
        model = va_utils.voas_cnn_model()
    elif (model_type == 5):
        ckpt_dir = './Checkpoints/voas_cnn_original.h5'
        model = va_utils.voas_cnn_model()
    va_utils.load_weights(model, ckpt_dir=ckpt_dir)

    mix_df = pd.DataFrame()
    sop_df = pd.DataFrame()
    alto_df = pd.DataFrame()
    ten_df = pd.DataFrame()
    bass_df = pd.DataFrame()

    for song in songlist:
        df_audio = pd.read_hdf('Datasets/Cantoria/Audio/Cantoria_' + song + '_Mix.h5', key='mix', mode='r')
        df_pyin_s = pd.read_csv('Datasets/Cantoria/F0_crepe/Cantoria_' + song + '_S.csv', header=None, engine='pyarrow')
        df_pyin_a = pd.read_csv('Datasets/Cantoria/F0_crepe/Cantoria_' + song + '_A.csv', header=None, engine='pyarrow')
        df_pyin_t = pd.read_csv('Datasets/Cantoria/F0_crepe/Cantoria_' + song + '_T.csv', header=None, engine='pyarrow')
        df_pyin_b = pd.read_csv('Datasets/Cantoria/F0_crepe/Cantoria_' + song + '_B.csv', header=None, engine='pyarrow')

        mix = df_audio.to_numpy().T
        raw_timescale = df_pyin_s[0].to_numpy()
        timescale = librosa.frames_to_time(frames = np.arange(df_audio.shape[0]), sr=22050, hop_length=256)

        splits = mix.shape[1]//256
        splits_diff = 256 - (mix.shape[1] - splits * 256)
        fill = np.zeros((360, splits_diff))
        mix_filled = np.concatenate((np.copy(mix), fill), axis=1)
        mix_filled = np.reshape(mix_filled, (360, -1, 256, 1)).transpose((1, 0, 2, 3))
        batches = math.ceil(mix_filled.shape[0]/24)

        s_pred = np.zeros((0, 360, 256))
        a_pred = np.zeros((0, 360, 256))
        t_pred = np.zeros((0, 360, 256))
        b_pred = np.zeros((0, 360, 256))

        for i in range(batches):
            s_pred_batch, a_pred_batch, t_pred_batch, b_pred_batch = model.predict(mix_filled[i*24:(i+1)*24])
            s_pred = np.append(s_pred, s_pred_batch, axis=0)
            a_pred = np.append(a_pred, a_pred_batch, axis=0)
            t_pred = np.append(t_pred, t_pred_batch, axis=0)
            b_pred = np.append(b_pred, b_pred_batch, axis=0)

        s_pred = va_utils.prediction_postproc(s_pred, high_pitch_fix=True)[:, :mix.shape[1]]
        a_pred = va_utils.prediction_postproc(a_pred, high_pitch_fix=True)[:, :mix.shape[1]]
        t_pred = va_utils.prediction_postproc(t_pred, high_pitch_fix=True)[:, :mix.shape[1]]
        b_pred = va_utils.prediction_postproc(b_pred, high_pitch_fix=True)[:, :mix.shape[1]]

        s_pred_freqs = va_utils.vec_bin_to_freq(np.argmax(s_pred, axis=0)).reshape(-1)
        a_pred_freqs = va_utils.vec_bin_to_freq(np.argmax(a_pred, axis=0)).reshape(-1)
        t_pred_freqs = va_utils.vec_bin_to_freq(np.argmax(t_pred, axis=0)).reshape(-1)
        b_pred_freqs = va_utils.vec_bin_to_freq(np.argmax(b_pred, axis=0)).reshape(-1)

        s_true_freqs = mir_eval.multipitch.resample_multipitch(raw_timescale, df_pyin_s[1].to_numpy().tolist(), timescale)
        a_true_freqs = mir_eval.multipitch.resample_multipitch(raw_timescale, df_pyin_a[1].to_numpy().tolist(), timescale)
        t_true_freqs = mir_eval.multipitch.resample_multipitch(raw_timescale, df_pyin_t[1].to_numpy().tolist(), timescale)
        b_true_freqs = mir_eval.multipitch.resample_multipitch(raw_timescale, df_pyin_b[1].to_numpy().tolist(), timescale)

        s_song_df = voice_metrics(s_true_freqs, s_pred_freqs, timescale)
        a_song_df = voice_metrics(a_true_freqs, a_pred_freqs, timescale)
        t_song_df = voice_metrics(t_true_freqs, t_pred_freqs, timescale)
        b_song_df = voice_metrics(b_true_freqs, b_pred_freqs, timescale)
        mix_song_df = mix_metrics(s_true_freqs, s_pred_freqs,
                                                a_true_freqs, a_pred_freqs,
                                                t_true_freqs, t_pred_freqs,
                                                b_true_freqs, b_pred_freqs,
                                                timescale)
        
        sop_df = pd.concat([sop_df, s_song_df], axis=0)
        alto_df = pd.concat([alto_df, a_song_df], axis=0)
        ten_df = pd.concat([ten_df, t_song_df], axis=0)
        bass_df = pd.concat([bass_df, b_song_df], axis=0)
        mix_df = pd.concat([mix_df, mix_song_df], axis=0)
        
        va_utils.song_to_midi(s_pred.T, a_pred.T, t_pred.T, b_pred.T, './MIDI/' + song + '.mid')

    sop_df.insert(loc=0, column='Songname', value=songlist)
    alto_df.insert(loc=0, column='Songname', value=songlist)
    ten_df.insert(loc=0, column='Songname', value=songlist)
    bass_df.insert(loc=0, column='Songname', value=songlist)
    mix_df.insert(loc=0, column='Songname', value=songlist) 

    sop_df = sop_df.set_index('Songname')
    alto_df = alto_df.set_index('Songname')
    ten_df = ten_df.set_index('Songname')
    bass_df = bass_df.set_index('Songname')
    mix_df = mix_df.set_index('Songname')

    return mix_df, sop_df, alto_df, ten_df, bass_df
