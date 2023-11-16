import va_utils
import va_plots
import cantoria_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def boxplot_generation(f_score_array: np.ndarray, title='', train='', dataset=''):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.boxplot(f_score_array.T)
    ax.set_ylim([0, 1])
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_xticklabels([f"Soprano\n{np.median(f_score_array[0]):.2f} ({np.std(f_score_array[0]):.2f})",
                        f"Contralto\n{np.median(f_score_array[1]):.2f} ({np.std(f_score_array[1]):.2f})",
                        f"Tenor\n{np.median(f_score_array[2]):.2f} ({np.std(f_score_array[2]):.2f})",
                        f"Baixo\n{np.median(f_score_array[3]):.2f} ({np.std(f_score_array[3]):.2f})"])
    if(title != ''):
        ax.set_title(title + ' (' + train + '/' + dataset + ')', pad=16, fontdict={'fontsize': 14})

    plt.savefig('../' + title + '_' + train + '_' + dataset + '.png', format='png', dpi=200)
    plt.close()

def evaluation_boxplots_generation(df_soprano: pd.DataFrame,
                        df_alto: pd.DataFrame,
                        df_tenor: pd.DataFrame,
                        df_bass: pd.DataFrame, title='', train='', dataset=''):

    precision = np.array([df_soprano['Precision'].to_numpy(),
                          df_alto['Precision'].to_numpy(),
                          df_tenor['Precision'].to_numpy(),
                          df_bass['Precision'].to_numpy()]).T
    
    recall = np.array([df_soprano['Recall'].to_numpy(),
                        df_alto['Recall'].to_numpy(),
                        df_tenor['Recall'].to_numpy(),
                        df_bass['Recall'].to_numpy()]).T
    
    f_score = np.array([df_soprano['F-Measure'].to_numpy(),
                        df_alto['F-Measure'].to_numpy(),
                        df_tenor['F-Measure'].to_numpy(),
                        df_bass['F-Measure'].to_numpy()]).T
    
    raw_pitch = np.array([df_soprano['Raw Pitch Accuracy'].to_numpy(),
                        df_alto['Raw Pitch Accuracy'].to_numpy(),
                        df_tenor['Raw Pitch Accuracy'].to_numpy(),
                        df_bass['Raw Pitch Accuracy'].to_numpy()]).T
    
    raw_chroma = np.array([df_soprano['Raw Chroma Accuracy'].to_numpy(),
                        df_alto['Raw Chroma Accuracy'].to_numpy(),
                        df_tenor['Raw Chroma Accuracy'].to_numpy(),
                        df_bass['Raw Chroma Accuracy'].to_numpy()]).T
    
    overall_acc = np.array([df_soprano['Overall Accuracy'].to_numpy(),
                        df_alto['Overall Accuracy'].to_numpy(),
                        df_tenor['Overall Accuracy'].to_numpy(),
                        df_bass['Overall Accuracy'].to_numpy()]).T

    fig, axs = plt.subplots(2, 3, figsize=(13, 8), dpi=200, constrained_layout=True)
    fig.set_layout_engine('tight')
    fig.subplots_adjust(left=0.15, bottom=0.925, right=0.175, top=1, wspace=0.2, hspace=0.2)
    if(title != ''):
        fig.suptitle(title + " (" + train + "/" + dataset + ")", fontsize=24)

    axs[0][0].yaxis.grid(True)
    axs[0][0].xaxis.grid(False)
    axs[0][0].boxplot(f_score)
    axs[0][0].set_title("F1-Score", pad=10)
    axs[0][0].set_ylim([0, 1])
    axs[0][0].set_xticklabels([ f"Soprano\n{np.median(f_score.T[0]):.2f}\n({np.std(f_score.T[0]):.2f})",
                                f"Alto\n{np.median(f_score.T[1]):.2f}\n({np.std(f_score.T[1]):.2f})",
                                f"Tenor\n{np.median(f_score.T[2]):.2f}\n({np.std(f_score.T[2]):.2f})",
                                f"Bass\n{np.median(f_score.T[3]):.2f}\n({np.std(f_score.T[3]):.2f})"])
    
    axs[0][1].yaxis.grid(True)
    axs[0][1].xaxis.grid(False)
    axs[0][1].boxplot(precision)
    axs[0][1].set_title("Precision", pad=10)
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_xticklabels([ f"Soprano\n{np.median(precision.T[0]):.2f}\n({np.std(precision.T[0]):.2f})",
                                f"Alto\n{np.median(precision.T[1]):.2f}\n({np.std(precision.T[1]):.2f})",
                                f"Tenor\n{np.median(precision.T[2]):.2f}\n({np.std(precision.T[2]):.2f})",
                                f"Bass\n{np.median(precision.T[3]):.2f}\n({np.std(precision.T[3]):.2f})"])
    
    axs[0][2].yaxis.grid(True)
    axs[0][2].xaxis.grid(False)
    axs[0][2].boxplot(recall)
    axs[0][2].set_title("Recall", pad=10)
    axs[0][2].set_ylim([0, 1])
    axs[0][2].set_xticklabels([ f"Soprano\n{np.median(recall.T[0]):.2f}\n({np.std(recall.T[0]):.2f})",
                                f"Alto\n{np.median(recall.T[1]):.2f}\n({np.std(recall.T[1]):.2f})",
                                f"Tenor\n{np.median(recall.T[2]):.2f}\n({np.std(recall.T[2]):.2f})",
                                f"Bass\n{np.median(recall.T[3]):.2f}\n({np.std(recall.T[3]):.2f})"])
    
    axs[1][0].yaxis.grid(True)
    axs[1][0].xaxis.grid(False)
    axs[1][0].boxplot(raw_pitch)
    axs[1][0].set_title("Raw Pitch Accuracy", pad=10)
    axs[1][0].set_ylim([0, 1])
    axs[1][0].set_xticklabels([ f"Soprano\n{np.median(raw_pitch.T[0]):.2f}\n({np.std(raw_pitch.T[0]):.2f})",
                                f"Alto\n{np.median(raw_pitch.T[1]):.2f}\n({np.std(raw_pitch.T[1]):.2f})",
                                f"Tenor\n{np.median(raw_pitch.T[2]):.2f}\n({np.std(raw_pitch.T[2]):.2f})",
                                f"Bass\n{np.median(raw_pitch.T[3]):.2f}\n({np.std(raw_pitch.T[3]):.2f})"])

    axs[1][1].yaxis.grid(True)
    axs[1][1].xaxis.grid(False)
    axs[1][1].boxplot(raw_chroma)
    axs[1][1].set_title("Raw Chroma Accuracy", pad=10)
    axs[1][1].set_ylim([0, 1])
    axs[1][1].set_xticklabels([ f"Soprano\n{np.median(raw_chroma.T[0]):.2f}\n({np.std(raw_chroma.T[0]):.2f})",
                                f"Alto\n{np.median(raw_chroma.T[1]):.2f}\n({np.std(raw_chroma.T[1]):.2f})",
                                f"Tenor\n{np.median(raw_chroma.T[2]):.2f}\n({np.std(raw_chroma.T[2]):.2f})",
                                f"Bass\n{np.median(raw_chroma.T[3]):.2f}\n({np.std(raw_chroma.T[3]):.2f})"])
    
    axs[1][2].yaxis.grid(True)
    axs[1][2].xaxis.grid(False)
    axs[1][2].boxplot(overall_acc)
    axs[1][2].set_title("Overall Accuracy", pad=10)
    axs[1][2].set_ylim([0, 1])
    axs[1][2].set_xticklabels([ f"Soprano\n{np.median(overall_acc.T[0]):.2f}\n({np.std(overall_acc.T[0]):.2f})",
                                f"Alto\n{np.median(overall_acc.T[1]):.2f}\n({np.std(overall_acc.T[1]):.2f})",
                                f"Tenor\n{np.median(overall_acc.T[2]):.2f}\n({np.std(overall_acc.T[2]):.2f})",
                                f"Bass\n{np.median(overall_acc.T[3]):.2f}\n({np.std(overall_acc.T[3]):.2f})"])

    plt.savefig('../Evaluation_' + title + '_' + train + '_' + dataset + '.png', format='png', dpi=200)
    plt.close()

def voicing_boxplots_generation(df_soprano: pd.DataFrame,
                     df_alto: pd.DataFrame,
                     df_tenor: pd.DataFrame,
                     df_bass: pd.DataFrame, title='', train='', dataset=''):

    voicing_recall = np.array([ df_soprano['Voicing Recall'].to_numpy(),
                                df_alto['Voicing Recall'].to_numpy(),
                                df_tenor['Voicing Recall'].to_numpy(),
                                df_bass['Voicing Recall'].to_numpy()]).T
    
    voicing_false = np.array([  df_soprano['Voicing False Alarm'].to_numpy(),
                                df_alto['Voicing False Alarm'].to_numpy(),
                                df_tenor['Voicing False Alarm'].to_numpy(),
                                df_bass['Voicing False Alarm'].to_numpy()]).T
    

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=200, constrained_layout=True)
    fig.set_layout_engine('tight')
    fig.subplots_adjust(left=0.15, bottom=0.925, right=0.175, top=1, wspace=0.2, hspace=0.2)
    if(title != ''):
        fig.suptitle(title + " (" + train + "/" + dataset + ")", fontsize=24)

    axs[0].yaxis.grid(True)
    axs[0].xaxis.grid(False)
    axs[0].boxplot(voicing_recall)
    axs[0].set_title("Voicing Recall", pad=10)
    axs[0].set_ylim([0, 1])
    axs[0].set_xticklabels([ f"Soprano\n{np.median(voicing_recall.T[0]):.2f}\n({np.std(voicing_recall.T[0]):.2f})",
                                f"Alto\n{np.median(voicing_recall.T[1]):.2f}\n({np.std(voicing_recall.T[1]):.2f})",
                                f"Tenor\n{np.median(voicing_recall.T[2]):.2f}\n({np.std(voicing_recall.T[2]):.2f})",
                                f"Bass\n{np.median(voicing_recall.T[3]):.2f}\n({np.std(voicing_recall.T[3]):.2f})"])
    
    axs[1].yaxis.grid(True)
    axs[1].xaxis.grid(False)
    axs[1].boxplot(voicing_false)
    axs[1].set_title("Voicing False Alarm", pad=10)
    axs[1].set_ylim([0, 1])
    axs[1].set_xticklabels([ f"Soprano\n{np.median(voicing_false.T[0]):.2f}\n({np.std(voicing_false.T[0]):.2f})",
                                f"Alto\n{np.median(voicing_false.T[1]):.2f}\n({np.std(voicing_false.T[1]):.2f})",
                                f"Tenor\n{np.median(voicing_false.T[2]):.2f}\n({np.std(voicing_false.T[2]):.2f})",
                                f"Bass\n{np.median(voicing_false.T[3]):.2f}\n({np.std(voicing_false.T[3]):.2f})"])
    
    plt.savefig('../Voicing_' + title + '_' + train + '_' + dataset + '.png', format='png', dpi=200)
    plt.close()
    
############################################################

for model in range(6):
    if (model == 0):
        log_folder = 'mask_voas_cnn'
        model_name = 'MaskVoasCNN'
    elif (model == 1):
        log_folder = 'mask_voas_v2'
        model_name = 'MaskVoasCNNv2'
    elif (model == 2):
        log_folder = 'downsample_voas_cnn'
        model_name = 'DownsampleVoasCNN'
    elif (model == 3):
        log_folder = 'downsample_voas_v2'
        model_name = 'DownsampleVoasCNNv2'
    elif (model == 4):
        log_folder = 'voas_cnn_retreino'
        model_name = 'VoasCNN (Retreino)'
    elif (model == 5):
        log_folder = 'voas_cnn_original'
        model_name = 'VoasCNN (Original)'

    metrics_dir = './Evaluation_Data/' + log_folder + '_f-scores_treino1.h5'
    mix_metrics, sop_metrics, alto_metrics, ten_metrics, bass_metrics = va_plots.metrics_load_precomputed(metrics_dir)
    holdout_sscs_f_scores = np.array([sop_metrics['F-Measure'].to_numpy(), alto_metrics['F-Measure'].to_numpy(), ten_metrics['F-Measure'].to_numpy(), bass_metrics['F-Measure'].to_numpy()])

    mix_df, sop_df, alto_df, ten_df, bass_df = cantoria_utils.compute_holdout_metrics(model)
    holdout_cantoria_f_scores = np.array([sop_df['F-Measure'].to_numpy(), alto_df['F-Measure'].to_numpy(), ten_df['F-Measure'].to_numpy(), bass_df['F-Measure'].to_numpy()])

    #boxplot_generation(holdout_sscs_f_scores, model_name, 'Holdout', 'SSCS')
    #boxplot_generation(holdout_cantoria_f_scores, model_name, 'Holdout', 'Cantoría')
    evaluation_boxplots_generation(sop_metrics, alto_metrics, ten_metrics, bass_metrics, model_name, 'Holdout', 'SSCS')
    evaluation_boxplots_generation(sop_df, alto_df, ten_df, bass_df, model_name, 'Holdout', 'Cantoría')
    voicing_boxplots_generation(sop_metrics, alto_metrics, ten_metrics, bass_metrics, model_name, 'Holdout', 'SSCS')
    voicing_boxplots_generation(sop_df, alto_df, ten_df, bass_df, model_name, 'Holdout', 'Cantoría')
