import itertools
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import pickle
import argparse

# first remove bacground than attach model detectors per image size
# each model use its color to mark detected bugs

from params import SAMPLE_SIZE, KERAS_MODEL_PATH, PATH_TEST_PHOTOS


def get_models():
    return (scaler, model, 10, 5, 0.005, 15)

def draw_square(pixels, x, y, probability):
    for d_x in range(-5, 5):
        for d_y in range(-5, 5):
            pixels[x + d_x, y + d_y] = (
                255,
                0,
                int(255 * probability))

def check_image(img,
                pixels,
                scaler,
                model,
                sample_size,
                step_size,
                convert_to_l=True,
                treshold=0.5,
                smaller_than_treshold=True,
                callback=None,
                draw_pixels=False,
                # Offset values for pixels
                o_x=0,
                o_y=0,
                ):
    image_width, image_height = img.size
    probabilities = set()

    for x in range(0, image_width - sample_size + 1, step_size):
        print 'analyzing row {}'.format(x)
        for y in range(0, image_height - sample_size + 1, step_size):
            cropped_img = img.crop((x, y, x + sample_size, y + sample_size))

            if convert_to_l:
                cropped_img = cropped_img.convert('L')

            arr_img = np.array(cropped_img)
            img_shape = arr_img.shape
            arr = scaler.transform(arr_img.reshape((1, -1))).reshape((1,) + img_shape + (1,))
            proba = model.predict(arr)[0]

            if (proba > treshold) == smaller_than_treshold:
                if callback:
                    print proba
                    callback(cropped_img, pixels, x + o_x, y + o_y)
                if draw_pixels:
                    probabilities.add(str(proba))
                    # Draw a colored square
                    draw_square(pixels, o_x + x + int(sample_size / 2), o_y + y + int(sample_size / 2), proba)

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
    parser.add_argument('--sample_size', type=int, default=64, help='the sample image is (SAMPLE_SIZE x SAMPLE_SIZE)')
    parser.add_argument('--text_overlap_sample_size', type=int, default=32, help='the sample of the overlapped text image')
    parser.add_argument('--text_step_size', type=int, default=16, help='the amount of pixels between every jump in text model')
    parser.add_argument('--text_overlap_step_size', type=int, default=1, help='the amount of pixels between every jump in text overlap model')
    parser.add_argument('--photos', type=str, default=PATH_TEST_PHOTOS, help='path of the test photo directory')
    parser.add_argument('--text_treshold', type=float, default=0.5, help='if predicted probability > treshold then is considered a match')
    parser.add_argument('--text_overlap_treshold', type=float, default=0.5, help='if predicted probability > treshold then is considered a match')
    FLAGS, unparsed = parser.parse_known_args()

    # load models
    text_overlap_model, text_overlap_scaler = load_model_and_scaler(FLAGS.text_overlap_model)
    text_model, text_scaler = load_model_and_scaler(FLAGS.text_model)

    #preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
    listing = os.listdir(FLAGS.photos)
    for photo in listing:
        try:
            img = Image.open(FLAGS.photos + '/' + photo)
            img_new = img.copy()
            pixels = img_new.load()
            # check_image(img,
            #     pixels,
            #     text_scaler,
            #     text_model,
            #     FLAGS.sample_size,
            #     FLAGS.text_step_size,
            #     treshold=FLAGS.text_treshold,
            #     callback=lambda cropped_img, pixels, x, y: check_image(
            #         cropped_img,
            #         pixels,
            #         text_overlap_scaler,
            #         text_overlap_model,
            #         FLAGS.text_overlap_sample_size,
            #         FLAGS.text_overlap_step_size,
            #         o_x=x,
            #         o_y=y,
            #         treshold=FLAGS.text_overlap_treshold,
            #         draw_pixels=True,
            #     )
            # )
            # Text NN
            # check_image(img,
            #     pixels,
            #     text_scaler,
            #     text_model,
            #     FLAGS.sample_size,
            #     FLAGS.text_step_size,
            #     treshold=FLAGS.text_treshold,
            #     draw_pixels=True,
            # )
            # Text overlap NN
            check_image(
                img,
                pixels,
                text_overlap_scaler,
                text_overlap_model,
                FLAGS.text_overlap_sample_size,
                FLAGS.text_overlap_step_size,
                treshold=FLAGS.text_overlap_treshold,
                draw_pixels=True,
            )
            img_new.show()
            img_new.save(FLAGS.photos + "/res/" + photo + "_t_" + '.png')
        except Exception as e:
            print e
