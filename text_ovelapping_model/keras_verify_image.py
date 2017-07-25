import itertools
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import pickle
import argparse

# first remove bacground than attach model detectors per image size
# each model use its color to mark detected bugs

from params import SAMPLE_SIZE, KERAS_MODEL_PATH, TRAINING_DATASET, PATH_TEST_PHOTOS


def get_models():
    return (scaler, model, 10, 5, 0.005, 15)


def check_image(img,
                pixels,
                scaler,
                model,
                sample_size,
                step_size,
                convert_to_l=True,
                treshold=0.5,
                mark_color=255,
                smaller_than_treshold=True
                ):
    image_width, image_height = img.size
    probabilities = set()

    for x in range(0, image_width - sample_size, step_size):
        print 'analyzing row {}'.format(x)
        for y in range(0, image_height - sample_size, step_size):
            cropped_img = img.crop((x, y, x + sample_size, y + sample_size))

            if convert_to_l:
                cropped_img = cropped_img.convert('L')

            arr_img = np.array(cropped_img)
            img_shape = arr_img.shape
            arr = scaler.transform(arr_img.reshape((1, -1))).reshape((1,) + img_shape + (1,))
            proba = model.predict(arr)[0]

            if (proba > treshold) == smaller_than_treshold:
                # Draw a colored square
                for d_x in range(-5, 5):
                    for d_y in range(-5, 5):
                        probabilities.add(str(proba))
                        pixels[x + d_x + int(sample_size / 2), y + d_y + int(sample_size / 2)] = (
                            255,
                            0,
                            int(255 * proba)
                        )
    print probabilities


def load_model_and_scaler(model_name):
    model = load_model(os.path.join(KERAS_MODEL_PATH, model_name + ".h5"))
    with open(os.path.join(KERAS_MODEL_PATH, model_name + '.pickle'), "rb") as f:
        scaler = pickle.load(f)
    return (model, scaler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_overlap_model', type=str, default='text_overlap', help='path of the overlapping model')
    parser.add_argument('--text_model', type=str, default='text', help='path of the text model')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE, help='the sample image is (SAMPLE_SIZE x SAMPLE_SIZE)')
    parser.add_argument('--step_size', type=int, default=SAMPLE_SIZE, help='the amount of pixels between every jump')
    parser.add_argument('--photos', type=str, default=PATH_TEST_PHOTOS, help='path of the test photo directory')
    FLAGS, unparsed = parser.parse_known_args()

    # load models
    text_overlap_model, text_overlap_scaler = load_model_and_scaler(FLAGS.text_overlap_model)
    # text_model = load_model_and_scaler(FLAGS.text_overlap_model)

    #preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
    listing = os.listdir(FLAGS.photos)
    for photo in listing:
        try:
            img = Image.open(FLAGS.photos + '/' + photo)
            img_new = img.copy()
            pixels = img_new.load()
            check_image(img, pixels, text_overlap_scaler, text_overlap_model, FLAGS.sample_size, FLAGS.step_size)
            img_new.show()
            img_new.save(FLAGS.photos + "/res/" + photo + "_t_" + '.png')
        except Exception as e:
            print e
