'''
eval_precompute_holdout.py
Developed by André Paiva - 2023

The eval_precompute script was created to automate the metrics evaluation
for the proposed models. The metrics are evaluated for the 805 songs of the test subset from
the SSCS Dataset. The script does the following steps:

-   Load each model and its correspondent weights;
-   Calculate the metrics with the function metrics_test_precompute()
    from the va_utils module;
-   Saves all the metrics on a .hdf5 file inside the
    "Evaluation_Data" directory.
'''

import va_utils

for i in range(6):
    if (i == 0):
        ckpt_dir = './Checkpoints/mask_voas_v2_treino1.keras'
        model = va_utils.mask_voas_cnn_v2_model()
        log_folder = 'mask_voas_v2'
    elif (i == 1):
        ckpt_dir = './Checkpoints/mask_voas_treino1.keras'
        model = va_utils.mask_voas_cnn_model()
        log_folder = 'mask_voas_cnn'
    elif (i == 2):
        ckpt_dir = './Checkpoints/downsample_voas_treino1.keras'
        model = va_utils.downsample_voas_cnn_model()
        log_folder = 'downsample_voas_cnn'
    elif (i == 3):
        ckpt_dir = './Checkpoints/downsample_voas_v2_treino1.keras'
        model = va_utils.downsample_voas_cnn_v2_model()
        log_folder = 'downsample_voas_v2'
    elif (i == 4):
        ckpt_dir = './Checkpoints/voas_treino1.keras'
        model = va_utils.voas_cnn_model()
        log_folder = 'voas_cnn_retreino'
    elif (i == 5):
        ckpt_dir = './Checkpoints/voas_cnn_original.h5'
        model = va_utils.voas_cnn_model()
        log_folder = 'voas_cnn_original'
        
    va_utils.load_weights(model, ckpt_dir=ckpt_dir)
    metrics_dir = './Evaluation_Data/' + log_folder + '_f-scores_treino1.h5'
    metrics = va_utils.metrics_test_precompute(model, metrics_dir)