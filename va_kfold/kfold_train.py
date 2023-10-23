import numpy as np
import tensorflow as tf
import va_utils

#Model Type:
#0 = VoasCNN (Retreino), 1 = VoasCNN (Original)
#2 = Downsample, 3 = Mask, 4 = MaskV2, 5 = DownsampleV2

TRAINING_NUMBER = 1
L_RATE = 5e-3
EPOCHS = 50

#################################

va_utils.download()

def kfold_train(fold_number):
    if fold_number == 1:
        ds_train = va_utils.get_dataset(split='train', start_index=800, end_index=4000)
        ds_val = va_utils.get_dataset(split='train', end_index=800)
    elif fold_number == 2:
        ds_train_a = va_utils.get_dataset(split='train', end_index=800)
        ds_train_b = va_utils.get_dataset(split='train', start_index=1600, end_index=4000)
        ds_train = ds_train_a.concatenate(ds_train_b)
        ds_val = va_utils.get_dataset(split='train', start_index=800, end_index=1600)
    elif fold_number == 3:
        ds_train_a = va_utils.get_dataset(split='train', end_index=1600)
        ds_train_b = va_utils.get_dataset(split='train', start_index=2400, end_index=4000)
        ds_train = ds_train_a.concatenate(ds_train_b)
        ds_val = va_utils.get_dataset(split='train', start_index=1600, end_index=2400)
    elif fold_number == 4:
        ds_train_a = va_utils.get_dataset(split='train', end_index=2400)
        ds_train_b = va_utils.get_dataset(split='train', start_index=3200, end_index=4000)
        ds_train = ds_train_a.concatenate(ds_train_b)
        ds_val = va_utils.get_dataset(split='train', start_index=2400, end_index=3200)
    elif fold_number == 5:
        ds_train = va_utils.get_dataset(split='train', end_index=3200)
        ds_val = va_utils.get_dataset(split='train', start_index=3200, end_index=4000)
    else:
        print('Invalid Fold Number.')

    ckpt_dir = './Checkpoints/mask_voas_v2_treino' + str(TRAINING_NUMBER) + '_kfold_' + str(fold_number) + '.keras'
    model = va_utils.mask_voas_cnn_v2_model(l_rate = L_RATE)
    log_folder = 'mask_voas_v2'

    metrics_dir = './Evaluation_Data/' + log_folder + '_treino' + str(TRAINING_NUMBER) + '_kfold_' + str(fold_number) + '.h5'

    va_utils.train(model, ds_train, ds_val, epochs=EPOCHS, save_model=True,
                ckpt_dir=ckpt_dir, log_folder=log_folder)

    metrics = va_utils.metrics_test_precompute(model, metrics_dir)