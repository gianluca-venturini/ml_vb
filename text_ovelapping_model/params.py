import os
sort = ['cropped-good100', 'cropped-white']
path = '../../../../ran/Documents/ML_bad_lf/overlap/'
model_path = path + "overlap_model.p"
keras_model_path = path + "kerasoverlap_model"
scaler_path = path + "scalar"
big_scaler_path = path + "big_scalar"
window_size = 100
step_size = 100
size = len([name for name in os.listdir(path + 'cropped')])
