import numpy as np
import tensorflow as tf
import va_utils

#Model Type:
#0 = VoasCNN (Retreino), 1 = VoasCNN (Original)
#2 = Downsample, 3 = Mask, 4 = MaskV2, 5 = DownsampleV2

MODEL_TYPE = 4
TRAINING_NUMBER = 2
L_RATE = 5e-3
EPOCHS = 30

#################################

va_utils.download()

ds_train = va_utils.get_dataset(split='train', end_index=1000)
ds_val = va_utils.get_dataset(split='validate', end_index=300)
ds_test = va_utils.get_dataset(split='test', end_index=300)

if (MODEL_TYPE == 0):
    ckpt_dir = './Checkpoints/voas_treino' + str(TRAINING_NUMBER) + '.keras'
    model = va_utils.voas_cnn_model(l_rate = L_RATE)
    log_folder = 'voas_cnn_retreino'
elif (MODEL_TYPE == 1):
    ckpt_dir = './Checkpoints/voas_cnn_original.h5'
    model = va_utils.voas_cnn_model(l_rate = L_RATE)
    log_folder = 'voas_cnn_original'
elif (MODEL_TYPE == 2):
    ckpt_dir = './Checkpoints/downsample_voas_treino' + str(TRAINING_NUMBER) + '.keras'
    model = va_utils.downsample_voas_cnn_model(l_rate = L_RATE)
    log_folder = 'downsample_voas_cnn'
elif (MODEL_TYPE == 3):
    ckpt_dir = './Checkpoints/mask_voas_treino' + str(TRAINING_NUMBER) + '.keras'
    model = va_utils.mask_voas_cnn_model(l_rate = L_RATE)
    log_folder = 'mask_voas_cnn'
elif (MODEL_TYPE == 4):
    ckpt_dir = './Checkpoints/mask_voas_v2_treino' + str(TRAINING_NUMBER) + '.keras'
    model = va_utils.mask_voas_cnn_v2_model(l_rate = L_RATE)
    log_folder = 'mask_voas_v2'
elif (MODEL_TYPE == 5):
    ckpt_dir = './Checkpoints/downsample_voas_v2_treino' + str(TRAINING_NUMBER) + '.keras'
    model = va_utils.downsample_voas_cnn_v2_model(l_rate = L_RATE)
    log_folder = 'downsample_voas_v2'
else:
    ckpt_dir = './Checkpoints/voas_cnn_original.h5'
    model = va_utils.voas_cnn_model(l_rate = L_RATE)
    log_folder = 'voas_cnn_original'

metrics_dir = './Evaluation_Data/' + log_folder + '_f-scores_treino' + str(TRAINING_NUMBER) + '.h5'

va_utils.train(model, ds_train, ds_val, epochs=EPOCHS, save_model=True,
               ckpt_dir=ckpt_dir, log_folder=log_folder)

metrics = va_utils.metrics_test_precompute(model, metrics_dir)