import os
import random
import itertools
import numpy
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
import pickle
from sklearn.preprocessing import StandardScaler
import argparse

from params import (
    KERAS_MODEL_PATH,
    SAMPLE_SIZE,
)

size = 0
INPUT_SIZE = SAMPLE_SIZE * SAMPLE_SIZE

def get_scaler(data_train):
    print 'Generate scaler'
    # We need to flatten the data in order to generate the scaler
    one_dim_data_train = [img.ravel() for img in data_train]
    scaler = StandardScaler().fit(one_dim_data_train)
    return scaler


def save_scaler(file_name, scaler):
    with open(file_name, "wb") as f:
        pickle.dump(scaler, f)


def scale_data(non_scaled_data, scaler):
    print 'Scaling the dataset'
    data_shape = non_scaled_data.shape
    # We need to flatten the data for scaling it
    reshaped_data = non_scaled_data.reshape((data_shape[0], -1))
    scaled_data = np.array(scaler.transform(reshaped_data))
    # Back to original shape
    return scaled_data.reshape(data_shape)


# preparing photo windows from given photos
def add_pcs(dataset_name, training_path, dataset_sample_size, dedup=True):
    images_and_labels = list()
    processed_image_hashes = set()
    for bucket in [0, 1]:
        bucket_training_path = os.path.join(training_path, dataset_name[bucket])
        print '[{}] dataset size: {}'.format(bucket, dataset_sample_size[bucket])
        file_list = os.listdir(bucket_training_path)
        dataset_size = len(file_list)
        print 'total dataset size: {}'.format(dataset_size)
        print dataset_name[bucket]
        file_sample_list = random.sample(file_list, dataset_sample_size[bucket])
        for photo in file_sample_list:
            if photo == ".DS_Store":
                continue
            try:
                img = Image.open(os.path.join(bucket_training_path, photo)).convert('L')
                # We create images that are (width x height x 1) because we have only one channel
                img_arr = np.array(img).reshape((img.size[0], img.size[1], 1))
                if dedup:
                    if hash(str(img_arr)) in processed_image_hashes:
                        continue

                    processed_image_hashes.add(hash(str(img_arr)))
                images_and_labels.append([img_arr, bucket])
            except Exception as e:
                pass
    return images_and_labels


def random_split(images_and_labels):
    random.shuffle(images_and_labels)
    size = len(images_and_labels)
    data = [x[0] for x in images_and_labels]
    label = [x[1] for x in images_and_labels]
    return (data, label, size)


def performance_measure(y_actual, y_hat):
    TN, FP, FN, TP = confusion_matrix(y_actual, y_hat).ravel()
    if TP + FN > 0:
        TPR = float(TP) / (TP + FN)
    else:
        TPR = -1
    if TN + FP > 0:
        TNR = float(TN) / (TN + FP)
    else:
        TNR = -1
    ACC = float(TP + TN) / (TP + FP + FN + TN)
    return ACC, TPR, TNR


def compute_TPR_TNR(y_test, y_predict):
    if np.array_equal(y_test, y_predict):
        return [1, 1, 1]
    TN, FP, FN, TP = confusion_matrix(y_test, y_predict).ravel()
    if TP + FN > 0:
        TPR = float(TP) / (TP + FN)
    else:
        TPR = -1
    if TN + FP > 0:
        TNR = float(TN) / (TN + FP)
    else:
        TNR = -1
    ACC = float(TP + TN) / (TP + FP + FN + TN)
    return [ACC, TPR, TNR]


s_model = None
old_TNR = 0
old_ACC = 0
old_TPR = 0
_k =0
_j=0

def get_convolution_model(layers):
    model = Sequential()
    model.add(Conv2D(layers[0], input_shape=(SAMPLE_SIZE, SAMPLE_SIZE, 1), activation='relu', kernel_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(layers[1], init='uniform', activation='relu'))
    model.add(Dense(layers[2], init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_train_and_test_data(images_and_labels):
    d, l, size = random_split(images_and_labels)
    data_train = np.array(d[:int(size * 0.9)])
    label_train = np.array(l[:int(size * 0.9)])
    data_test = np.array(d[int(size * 0.1):])
    label_test = np.array(l[int(size * 0.1):])
    return (data_train, label_train, data_test, label_test)


def train(model, epochs=50):

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data_train, label_train, validation_split=0.02, epochs=epochs)
    predictions = model.predict(data_test)
    # round predictions
    rounded_predictions = [round(x[0]) for x in predictions]
    [ACC, TPR, TNR] = compute_TPR_TNR(label_test, rounded_predictions)

    return (ACC, TPR, TNR)

def print_statistics(ACC, TPR, TNR):
    print 'Accuracy is: ' + str(ACC) + ' True Positive Rate: ' + str(
            TPR) + ' True Negative Rate: ' + str(TNR)  + " "+ str(_j) + " k: " +str(_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=str, required=True, help='the path for the training dataset')
    parser.add_argument('--model_name', type=str, required=True, help='the name of the output model')
    parser.add_argument('--num_good_train_images', type=int, default=100, help='number of images to sample from the good dataset')
    parser.add_argument('--num_bug_train_images', type=int, default=100, help='number of images to sample from the bug dataset')
    parser.add_argument('--dedup', type=bool, default=True, help='if True the good and bad sampled datasets are disjoint')
    parser.add_argument('--dataset_good_name', type=str, default='good', help='name of the directory that contains good images')
    parser.add_argument('--dataset_bug_name', type=str, default='bug', help='name of directory that contains bug images')
    FLAGS, unparsed = parser.parse_known_args()

    numpy.random.seed()

    dataset_sample_size=[FLAGS.num_good_train_images, FLAGS.num_bug_train_images]
    images_and_labels = add_pcs([FLAGS.dataset_good_name, FLAGS.dataset_bug_name], FLAGS.training, dataset_sample_size, dedup=FLAGS.dedup)
    print len(images_and_labels)
    print "=============================="
    data_train, label_train, data_test, label_test = get_train_and_test_data(images_and_labels)
    # Scale the data in order to have mean=0 and variance=1
    scaler = get_scaler(data_train)
    data_train = scale_data(data_train, scaler)
    data_test = scale_data(data_test, scaler)
    save_scaler(KERAS_MODEL_PATH + FLAGS.model_name + '.pickle', scaler)
    model = get_convolution_model(layers=[25, 25, 10])
    ACC, TPR, TNR = train(model, epochs=15)
    model.save(KERAS_MODEL_PATH + FLAGS.model_name + ".h5")
    print_statistics(ACC, TPR, TNR)
