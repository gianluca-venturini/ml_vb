import os
import itertools


import numpy as np
from keras.models import load_model
from PIL import Image
import os
import pickle

# first remove bacground than attach model detectors per image size
# each model use its color to mark detected bugs

path = '/Users/ran/Documents/ML_bad_lf/indent/'
path_photo  = path +"test/indent/bad/"
scaler_path = path+"scaler"
model_name = 'cropped-text12'
keras_model_path = path+"model"+model_name+".h5"
#
# window_size = 12
# step_size = 5

# load models
background_model = load_model(keras_model_path)
with open(scaler_path + model_name, "rb") as f:
    background_scaler = pickle.load(f)

inner_model_name = "cropped-overlap-10"
with open(scaler_path + inner_model_name, "rb") as f:
    scaler_10 = pickle.load(f)
model_10 = load_model(path+"model" + inner_model_name + ".h5")


def get_models(window_size):
    if window_size == 0:
        return (scaler_10, model_10, 10, 1, 0.9999999, 15, True)


def check_image(img, pixels,
                scaler, model,
                window_size,
                step_size,
                x, y,
                convert_to_l=True,
                tresh_hold=0.99,
                mark_color=30,
                smaller_than_treshhold=True,
                ):
    n, m = img.size
    s = set()
    colored = []
    for j, i in itertools.product(range(0, m - window_size, step_size),
                                  range(0, n - window_size, step_size)):
        print '.',

        cropped_img = img.crop((i, j, i + window_size, j + window_size))
        if convert_to_l:
            cropped_img = cropped_img.convert('L')

        arr_orig = np.array(cropped_img).ravel()

        arr = scaler.transform(arr_orig).tolist()
        proba = model.predict(np.array([arr]))[0]
        r = window_size/2
        if (proba > tresh_hold)== smaller_than_treshhold:
            colored.append((i,j))
            s.add(str(proba))
            for k, l in itertools.product(range(-2*r, 2*r), range(-r, r)):
                if mark_color is not 0:
                    pixels[x + i + int(window_size / 2) + k, y + j + int(window_size / 2) + l] = (
                255, mark_color, int(200 * proba))
            # get scalar and model per image size
            params = get_models(window_size)
            if params:
                sc, mo, win_s, step_s, t_hold, m_color, smtt= params
                check_image(cropped_img, pixels, sc, mo, win_s, step_s, x + i, y + j,
                            convert_to_l=False,
                            tresh_hold=t_hold,
                            mark_color=100,
                            smaller_than_treshhold=smtt
                )

    return remove_non_relevant_points_inside_rect(colored, step_size+window_size)


# leave only edge points of colored lines
def remove_non_relevant_points_inside_rect(line, window_size):
    space = max([window_size, 27])
    min_sentence = 35
    rects = []
    if not line:
        return
    x, y = line[0]
    d = window_size
    for a, b in line:
        if b == y and (a - x) < (space + d):
            d = a-x
        else:
            if d > min_sentence:
                rects.append((x, y,d))
            (x, y, d) = (a,b, window_size)
    return rects

def paint_by_list(rects, window_size, pixels):
    mark_color =  3
    for x, y , d in rects:
        for k, l in itertools.product(range(d), range(window_size)):
            pixels[x + k, y + l] = (
                15, mark_color, 200)


#preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
listing = os.listdir(path_photo)
length = []
for photo in listing:
    try:
        img = Image.open(path_photo + '/' + photo)
        img_new = img.copy()
        pixels = img_new.load()
        c_list = check_image(img, pixels, background_scaler, background_model, 12, 12, 0, 0)
        with open(path_photo+ "/res-nr/"+ "nr_" + photo[:-2], "wb") as f:
            pickle.dump(c_list, f)
        paint_by_list(c_list, 12, pixels)
        print "|||||||||||||||||||||||||"
        length.append(len(c_list))
        print "|||||||||||||||||||||||||"
        img_new.show()
        img_new.save(path_photo + "/res-img/" + photo + "_t_" + '.png')
    except IOError as e:
        # raise
        print e
print length

