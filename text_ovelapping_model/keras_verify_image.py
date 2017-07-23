import itertools


import numpy as np
from keras.models import load_model
from PIL import Image
import os
import pickle

# first remove bacground than attach model detectors per image size
# each model use its color to mark detected bugs

from params import SAMPLE_SIZE, STEP_SIZE, KERAS_MODEL_PATH, SCALER_TRAINING_FILE, TRAINING_DATASET

path_photo = '../test'


# load models
background_model = load_model(KERAS_MODEL_PATH + TRAINING_DATASET[1] + ".h5")
with open(SCALER_TRAINING_FILE, "rb") as f:
    scaler = pickle.load(f)
model = load_model(KERAS_MODEL_PATH + TRAINING_DATASET[1] + ".h5")


def get_models():
    return (scaler, model, 10, 5, 0.005, 15)


def check_image(img,
                pixels,
                scaler,
                model,
                sample_size,
                step_size,
                convert_to_l=True,
                tresh_hold=0.5,
                mark_color=255,
                smaller_than_treshold=True
                ):
    image_width, image_height = img.size
    probabilities = set()
    # s = set()
    # for i, j in itertools.product(range(0, n - image_size, step_size),
    #                               range(0, m - image_size, step_size)):
    #     print '.',

    #     cropped_img = img.crop((i, j, i + image_size, j + image_size))
    #     if convert_to_l:
    #         cropped_img = cropped_img.convert('L')

    #     arr_orig = np.array(cropped_img).ravel()

    #     arr = scaler.transform(arr_orig).tolist()
    #     proba = model.predict(np.array([arr]))[0]
    #     if (proba > tresh_hold)== smaller_than_treshold:
    #         for k, l in itertools.product(range(-5, 5), range(-5, 5)):
    #             s.add(str(proba))
    #             pixels[x + i + int(image_size / 2) + k, y + j + int(image_size / 2) + l] = (
    #             255, mark_color, int(200 * proba))
    #         # get scalar and model per image size
    #         params = get_models(image_size)
    #         if params:
    #             sc, mo, win_s, step_s, t_hold, m_color = params
    #             check_image(cropped_img, pixels, sc, mo, win_s, step_s, x + i, y + j,
    #                         convert_to_l=False,
    #                         tresh_hold=t_hold,
    #                         mark_color=m_color)
    # print s

    for x in range(0, image_width - sample_size):
        print 'analyzing row {}'.format(x)
        for y in range(0, image_height - sample_size):
            print '{}.{}'.format(x, y)
            cropped_img = img.crop((x, y, x + sample_size, y + sample_size))

            if convert_to_l:
                cropped_img = cropped_img.convert('L')

            arr_orig = np.array(cropped_img).ravel()
            arr = scaler.transform(arr_orig).tolist()
            proba = model.predict(np.array([arr]))[0]

            if (proba > tresh_hold)== smaller_than_treshold:
                # Draw a colored square
                for d_x in range(-5, 5):
                    for d_y in range(-5, 5):
                        probabilities.add(str(proba))
                        pixels[x + d_x + int(sample_size / 2), y + d_y + int(sample_size / 2)] = (
                            255,
                            mark_color,
                            int(200 * proba)
                        )
    print probabilities




#preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
listing = os.listdir(path_photo)
for photo in listing:
    try:
        img = Image.open(path_photo + '/' + photo)
        img_new = img.copy()
        pixels = img_new.load()
        check_image(img, pixels, scaler, background_model, SAMPLE_SIZE, STEP_SIZE)
        img_new.show()
        img_new.save(path_photo + "/res/" + photo + "_t_" + '.png')
    except Exception as e:
        print e

