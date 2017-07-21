import itertools


import numpy as np
from keras.models import load_model
from PIL import Image
import os
import pickle

# first remove bacground than attach model detectors per image size
# each model use its color to mark detected bugs

from text_ovelapping_model.params import window_size, step_size, keras_model_path, scaler_path, big_scaler_path, sort

path_photo = '../../../../ran/Documents/ML_bad_lf/overlap/test'


# load models
background_model = load_model(keras_model_path + sort[1] + ".h5")
with open(scaler_path + sort[1], "rb") as f:
    background_scaler = pickle.load(f)
model_name = "cropped-10"
with open(scaler_path + model_name, "rb") as f:
    scaler_10 = pickle.load(f)
model_10 = load_model(keras_model_path + model_name + ".h5")


def get_models(window_size):
    if window_size == 100:
        return (scaler_10, model_10, 10, 5, 0.005, 15)


def check_image(img, pixels,
                scaler, model,
                window_size,
                step_size,
                x, y,
                convert_to_l=True,
                tresh_hold=0.5,
                mark_color=255,
                smaller_than_treshhold=True
                ):
    n, m = img.size
    s = set()
    for i, j in itertools.product(range(0, n - window_size, step_size),
                                  range(0, m - window_size, step_size)):
        print '.',

        cropped_img = img.crop((i, j, i + window_size, j + window_size))
        if convert_to_l:
            cropped_img = cropped_img.convert('L')

        arr_orig = np.array(cropped_img).ravel()

        arr = scaler.transform(arr_orig).tolist()
        proba = model.predict(np.array([arr]))[0]
        if (proba > tresh_hold)== smaller_than_treshhold:
            for k, l in itertools.product(range(-5, 5), range(-5, 5)):
                s.add(str(proba))
                pixels[x + i + int(window_size / 2) + k, y + j + int(window_size / 2) + l] = (
                255, mark_color, int(200 * proba))
            # get scalar and model per image size
            params = get_models(window_size)
            if params:
                sc, mo, win_s, step_s, t_hold, m_color = params
                check_image(cropped_img, pixels, sc, mo, win_s, step_s, x + i, y + j,
                            convert_to_l=False,
                            tresh_hold=t_hold,
                            mark_color=m_color)

    print s


#preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
listing = os.listdir(path_photo)
for photo in listing:
    try:
        img = Image.open(path_photo + '/' + photo)
        img_new = img.copy()
        pixels = img_new.load()
        check_image(img, pixels, background_scaler, background_model, window_size, step_size, 0, 0)
        img_new.show()
        img_new.save(path_photo + "/res/" + photo + "_t_" + '.png')
    except Exception as e:
        print e
