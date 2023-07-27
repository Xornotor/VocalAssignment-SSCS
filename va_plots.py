import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

font_dirs = './Assets/Fonts/'
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font in font_files: fm.fontManager.addfont(font)

plt.rcParams['font.family'] = "SF UI Text"
plt.rcParams['font.size'] = 14

############################################################

def metrics_load_precomputed(file_path: str):
    """Loads precomputed evaluation metrics in HDF5. 

    Parameters
    ----------
    ``file_path`` : String
        HDF5 File Path.

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
    mix_df = pd.read_hdf(file_path, key='mix', mode='r')
    sop_df = pd.read_hdf(file_path, key='soprano', mode='r')
    alto_df = pd.read_hdf(file_path, key='alto', mode='r')
    ten_df = pd.read_hdf(file_path, key='tenor', mode='r')
    bass_df = pd.read_hdf(file_path, key='bass', mode='r')
    return mix_df, sop_df, alto_df, ten_df, bass_df

############################################################

def boxplot(f_score_array: np.ndarray, title=''):    
    """Plots a F-Score boxplot for Soprano, Alto, Tenor and Bass using
    ``matplotlib``.

    Parameters
    ----------
    ``f_score_array`` : np.ndarray
        Array of shape (4, N).

        f_score_array[0]: Soprano F-Scores

        f_score_array[1]: Alto F-Scores

        f_score_array[2]: Tenor F-Scores

        f_score_array[3]: Bass F-Scores

    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """
    fig, ax = plt.subplots(figsize=(4, 6), dpi=200)
    ax.boxplot(f_score_array.T)
    ax.set_ylim([0, 1])
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_xticklabels([f"Soprano\n{np.median(f_score_array[0]):.2f}",
                        f"Alto\n{np.median(f_score_array[1]):.2f}",
                        f"Tenor\n{np.median(f_score_array[2]):.2f}",
                        f"Bass\n{np.median(f_score_array[3]):.2f}"])
    if(title != ''):
        ax.set_title(title)

    plt.show()

############################################################

def joint_f_histograms(f_scores: np.ndarray, title=''):
    """Plots a F-Score multiplot combining the histograms of four voices
     (Soprano, Alto, Tenor and Bass) using ``matplotlib``.

    Parameters
    ----------
    ``f_scores`` : np.ndarray
        Array of shape (4, N).

        f_scores[0]: Soprano F-Scores

        f_scores[1]: Alto F-Scores

        f_scores[2]: Tenor F-Scores

        f_scores[3]: Bass F-Scores
    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """
    s_counts, s_bins = np.histogram(f_scores[0], bins=100)
    a_counts, a_bins = np.histogram(f_scores[1], bins=100)
    t_counts, t_bins = np.histogram(f_scores[2], bins=100)
    b_counts, b_bins = np.histogram(f_scores[3], bins=100)
    plt.figure(figsize=(12,4.5), dpi=200)
    plt.grid(visible=True, axis='y')
    plt.stairs(s_counts, s_bins, label='soprano')
    plt.stairs(a_counts, a_bins, label='alto')
    plt.stairs(t_counts, t_bins, label='tenor')
    plt.stairs(b_counts, b_bins, label='bass')
    plt.ylim(0, 225)
    plt.legend()
    if(title != ''):
        plt.title(title)
    plt.show()

############################################################

def voice_f_histograms(f_scores: np.ndarray, title=''):
    """Generates a subplot imagem with four histograms for the F-Scores of four voices
     (Soprano, Alto, Tenor and Bass) using ``matplotlib``.

    Parameters
    ----------
    ``f_scores`` : np.ndarray
        Array of shape (4, N).

        f_scores[0]: Soprano F-Scores

        f_scores[1]: Alto F-Scores

        f_scores[2]: Tenor F-Scores

        f_scores[3]: Bass F-Scores

    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """
    s_counts, s_bins = np.histogram(f_scores[0], bins=100)
    a_counts, a_bins = np.histogram(f_scores[1], bins=100)
    t_counts, t_bins = np.histogram(f_scores[2], bins=100)
    b_counts, b_bins = np.histogram(f_scores[3], bins=100)

    fig, axs = plt.subplots(2, 2, figsize=(15, 7), dpi=200, constrained_layout=True)
    fig.set_layout_engine('tight')
    fig.subplots_adjust(left=0.15, bottom=0.925, right=0.175, top=1, wspace=0.2, hspace=0.2)
    if(title != ''):
        fig.suptitle(title, fontsize=30)
    axs[0][0].yaxis.grid(True)
    axs[0][0].xaxis.grid(False)
    axs[0][0].stairs(s_counts, s_bins, fill=True)
    axs[0][0].set_title("Soprano - Histograma")
    axs[0][0].set_xlim([0, 1])
    axs[0][0].set_ylim([0, 225])

    axs[0][1].yaxis.grid(True)
    axs[0][1].xaxis.grid(False)
    axs[0][1].stairs(a_counts, a_bins, fill=True, color='orange')
    axs[0][1].set_title("Alto - Histograma")
    axs[0][1].set_xlim([0, 1])
    axs[0][1].set_ylim([0, 225])
    
    axs[1][0].yaxis.grid(True)
    axs[1][0].xaxis.grid(False)
    axs[1][0].stairs(t_counts, t_bins, fill=True, color='green')
    axs[1][0].set_title("Tenor - Histograma")
    axs[1][0].set_xlim([0, 1])
    axs[1][0].set_ylim([0, 225])
    
    axs[1][1].yaxis.grid(True)
    axs[1][1].xaxis.grid(False)
    axs[1][1].stairs(b_counts, b_bins, fill=True, color='red')
    axs[1][1].set_title("Bass - Histograma")
    axs[1][1].set_xlim([0, 1])
    axs[1][1].set_ylim([0, 225])
    
    plt.show()

############################################################

def plot(dataframe: np.ndarray, colorbar=False, title=''):
    """Plots a heatmap using ``matplotlib``. Used for pitch salience plots.

    Parameters
    ----------
    ``dataframe`` : np.ndarray
        Array of shape (360, N).
    ``colorbar`` : boolean
        Enable/disable colorbar. Default is ``False``.
    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """
    aspect_ratio = (3/8)*dataframe.shape[1]/dataframe.shape[0]
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    if(title != ''):
        ax.set_title(title)
    im = ax.imshow(dataframe, interpolation='nearest', aspect=aspect_ratio,
        cmap = mpl.colormaps['BuPu'])
    if colorbar:
        fig.colorbar(im, shrink=0.5)
    ax.invert_yaxis()
    plt.show()

############################################################

def plot_activation_maps(actv_maps: np.ndarray, colorbar=False, title=''):
    """Plots activation maps using ``matplotlib``. Used for plotting 
    activation maps for convolutional layers with 16 filters.

    Parameters
    ----------
    ``actv_maps`` : np.ndarray
        Array of shape (360, N, 16).
    ``colorbar`` : boolean
        Enable/disable colorbar. Default is ``False``.
    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """

    aspect_ratio = (3.75/8)*actv_maps.shape[1]/actv_maps.shape[0]
    fig, axs = plt.subplots(4, 4, figsize=(13, 6), dpi=500, constrained_layout=True)
    if(title != ''):
        fig.suptitle(title)
    for i in range(4):
        for j in range(4):
            im = axs[i][j].imshow(actv_maps[:, :, 4*i + j], interpolation='nearest',
                            aspect=aspect_ratio, cmap = mpl.colormaps['BuPu'])
            axs[i][j].invert_yaxis()
            axs[i][j].xaxis.set_tick_params(labelbottom=False)
            axs[i][j].yaxis.set_tick_params(labelleft=False)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=axs[:, :], shrink=0.7, location='right')
    plt.show()

############################################################

def evaluation_boxplots(df_soprano: pd.DataFrame,
                        df_alto: pd.DataFrame,
                        df_tenor: pd.DataFrame,
                        df_bass: pd.DataFrame, title=''):
    
    """Plots evaluation metrics boxplots using ``matplotlib``.

    Parameters
    ----------
    ``df_soprano`` : np.ndarray
        Array of shape (360, N, 16).
    ``colorbar`` : boolean
        Enable/disable colorbar. Default is ``False``.
    ``title``: String
        Title for the plot

    Returns
    -------
    ``None``
    """

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
        fig.suptitle(title, fontsize=30)

    axs[0][0].yaxis.grid(True)
    axs[0][0].xaxis.grid(False)
    axs[0][0].boxplot(f_score)
    axs[0][0].set_title("F-Score")
    axs[0][0].set_ylim([0, 1])
    axs[0][0].set_xticklabels([ f"Soprano\n{np.median(f_score.T[0]):.2f}\n({np.std(f_score.T[0]):.2f})",
                                f"Alto\n{np.median(f_score.T[1]):.2f}\n({np.std(f_score.T[1]):.2f})",
                                f"Tenor\n{np.median(f_score.T[2]):.2f}\n({np.std(f_score.T[2]):.2f})",
                                f"Bass\n{np.median(f_score.T[3]):.2f}\n({np.std(f_score.T[3]):.2f})"])
    
    axs[0][1].yaxis.grid(True)
    axs[0][1].xaxis.grid(False)
    axs[0][1].boxplot(precision)
    axs[0][1].set_title("Precision")
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_xticklabels([ f"Soprano\n{np.median(precision.T[0]):.2f}\n({np.std(precision.T[0]):.2f})",
                                f"Alto\n{np.median(precision.T[1]):.2f}\n({np.std(precision.T[1]):.2f})",
                                f"Tenor\n{np.median(precision.T[2]):.2f}\n({np.std(precision.T[2]):.2f})",
                                f"Bass\n{np.median(precision.T[3]):.2f}\n({np.std(precision.T[3]):.2f})"])
    
    axs[0][2].yaxis.grid(True)
    axs[0][2].xaxis.grid(False)
    axs[0][2].boxplot(recall)
    axs[0][2].set_title("Recall")
    axs[0][2].set_ylim([0, 1])
    axs[0][2].set_xticklabels([ f"Soprano\n{np.median(recall.T[0]):.2f}\n({np.std(recall.T[0]):.2f})",
                                f"Alto\n{np.median(recall.T[1]):.2f}\n({np.std(recall.T[1]):.2f})",
                                f"Tenor\n{np.median(recall.T[2]):.2f}\n({np.std(recall.T[2]):.2f})",
                                f"Bass\n{np.median(recall.T[3]):.2f}\n({np.std(recall.T[3]):.2f})"])
    
    axs[1][0].yaxis.grid(True)
    axs[1][0].xaxis.grid(False)
    axs[1][0].boxplot(raw_pitch)
    axs[1][0].set_title("Raw Pitch Accuracy")
    axs[1][0].set_ylim([0, 1])
    axs[1][0].set_xticklabels([ f"Soprano\n{np.median(raw_pitch.T[0]):.2f}\n({np.std(raw_pitch.T[0]):.2f})",
                                f"Alto\n{np.median(raw_pitch.T[1]):.2f}\n({np.std(raw_pitch.T[1]):.2f})",
                                f"Tenor\n{np.median(raw_pitch.T[2]):.2f}\n({np.std(raw_pitch.T[2]):.2f})",
                                f"Bass\n{np.median(raw_pitch.T[3]):.2f}\n({np.std(raw_pitch.T[3]):.2f})"])

    axs[1][1].yaxis.grid(True)
    axs[1][1].xaxis.grid(False)
    axs[1][1].boxplot(raw_chroma)
    axs[1][1].set_title("Raw Chroma Accuracy")
    axs[1][1].set_ylim([0, 1])
    axs[1][1].set_xticklabels([ f"Soprano\n{np.median(raw_chroma.T[0]):.2f}\n({np.std(raw_chroma.T[0]):.2f})",
                                f"Alto\n{np.median(raw_chroma.T[1]):.2f}\n({np.std(raw_chroma.T[1]):.2f})",
                                f"Tenor\n{np.median(raw_chroma.T[2]):.2f}\n({np.std(raw_chroma.T[2]):.2f})",
                                f"Bass\n{np.median(raw_chroma.T[3]):.2f}\n({np.std(raw_chroma.T[3]):.2f})"])
    
    axs[1][2].yaxis.grid(True)
    axs[1][2].xaxis.grid(False)
    axs[1][2].boxplot(overall_acc)
    axs[1][2].set_title("Overall Accuracy")
    axs[1][2].set_ylim([0, 1])
    axs[1][2].set_xticklabels([ f"Soprano\n{np.median(overall_acc.T[0]):.2f}\n({np.std(overall_acc.T[0]):.2f})",
                                f"Alto\n{np.median(overall_acc.T[1]):.2f}\n({np.std(overall_acc.T[1]):.2f})",
                                f"Tenor\n{np.median(overall_acc.T[2]):.2f}\n({np.std(overall_acc.T[2]):.2f})",
                                f"Bass\n{np.median(overall_acc.T[3]):.2f}\n({np.std(overall_acc.T[3]):.2f})"])

    plt.show()

############################################################
