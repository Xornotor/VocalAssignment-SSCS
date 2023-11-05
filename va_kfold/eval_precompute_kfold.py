'''
eval_precompute_kfold.py
Developed by André Paiva - 2023
'''

import va_utils

for i in range(1, 6):
    model = va_utils.mask_voas_cnn_v2_model()
    log_folder = 'mask_voas_v2'
    ckpt_dir = './Checkpoints/mask_voas_v2_treino2_kfold_' + str(i) + '.keras'        
    va_utils.load_weights(model, ckpt_dir=ckpt_dir)
    metrics_dir = './Evaluation_Data/' + log_folder + '_treino2_kfold_' + str(i) + '.h5'
    #songlist = va_utils.pick_songlist(first = 800*(i-1), amount=800, split='train')
    #metrics = va_utils.metrics_val_precompute(model, metrics_dir, songlist)
    metrics = va_utils.metrics_test_precompute(model, metrics_dir)