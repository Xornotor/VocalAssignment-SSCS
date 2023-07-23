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

def metrics_load_precomputed(file_path):
    mix_df = pd.read_hdf(file_path, key='mix', mode='r')
    sop_df = pd.read_hdf(file_path, key='soprano', mode='r')
    alto_df = pd.read_hdf(file_path, key='alto', mode='r')
    ten_df = pd.read_hdf(file_path, key='tenor', mode='r')
    bass_df = pd.read_hdf(file_path, key='bass', mode='r')
    return mix_df, sop_df, alto_df, ten_df, bass_df

############################################################

def boxplot(f_score_array, title=''):    
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

def joint_f_histograms(f_scores, title=''):
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
    plt.ylim(0, 120)
    plt.legend()
    if(title != ''):
        plt.title(title)
    plt.show()

############################################################

def voice_f_histograms(f_scores, title=''):
    s_counts, s_bins = np.histogram(f_scores[0], bins=100)
    a_counts, a_bins = np.histogram(f_scores[1], bins=100)
    t_counts, t_bins = np.histogram(f_scores[2], bins=100)
    b_counts, b_bins = np.histogram(f_scores[3], bins=100)

    fig, axs = plt.subplots(2, 2, figsize=(15, 7), dpi=200, constrained_layout=True)
    fig.subplots_adjust(top=0.85)
    #fig.tight_layout(pad=2.0)
    if(title != ''):
        fig.suptitle(title)
    axs[0][0].yaxis.grid(True)
    axs[0][0].xaxis.grid(False)
    axs[0][0].stairs(s_counts, s_bins, fill=True)
    axs[0][0].set_title("Soprano - Histograma")
    axs[0][0].set_xlim([0, 1])
    axs[0][0].set_ylim([0, 120])

    axs[0][1].yaxis.grid(True)
    axs[0][1].xaxis.grid(False)
    axs[0][1].stairs(a_counts, a_bins, fill=True, color='orange')
    axs[0][1].set_title("Alto - Histograma")
    axs[0][1].set_xlim([0, 1])
    axs[0][1].set_ylim([0, 120])
    
    axs[1][0].yaxis.grid(True)
    axs[1][0].xaxis.grid(False)
    axs[1][0].stairs(t_counts, t_bins, fill=True, color='green')
    axs[1][0].set_title("Tenor - Histograma")
    axs[1][0].set_xlim([0, 1])
    axs[1][0].set_ylim([0, 120])
    
    axs[1][1].yaxis.grid(True)
    axs[1][1].xaxis.grid(False)
    axs[1][1].stairs(b_counts, b_bins, fill=True, color='red')
    axs[1][1].set_title("Bass - Histograma")
    axs[1][1].set_xlim([0, 1])
    axs[1][1].set_ylim([0, 120])
    
    plt.show()

############################################################

def plot(dataframe, colorbar=False, title=''):

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

def plot_activation_maps(actv_maps, colorbar=False, title=''):

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
