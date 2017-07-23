import os
import random
import itertools
import numpy
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
import pickle
from sklearn.preprocessing import StandardScaler

from params import (
    TRAINING_PATH,
    SAMPLE_SIZE,
    SCALER_TRAINING_FILE,
    KERAS_MODEL_PATH,
    TRAINING_DATASET,
    SAMPLE_SIZE,
)

size = 0
INPUT_SIZE = SAMPLE_SIZE * SAMPLE_SIZE

def get_scaler(data_train):
    scaler = StandardScaler().fit(data_train)
    return scaler

def save_scaler(scaler):
    with open(SCALER_TRAINING_FILE, "wb") as f:
        pickle.dump(scaler, f)


def scale_data(non_scaled_data, scaler):
    print non_scaled_data

    return  scaler.transform(non_scaled_data).tolist()

# preparing photo windows from given photos
def add_pcs(dedup=True):
    images_and_labels = list()
    processed_image_hashes = set()
    for label in [0, 1]:
        print SAMPLE_SIZE[label]
        print len(os.listdir(TRAINING_PATH + TRAINING_DATASET[label]))
        print TRAINING_DATASET[label]
        listing = random.sample(os.listdir(TRAINING_PATH + TRAINING_DATASET[label]), SAMPLE_SIZE[label])
        for photo in listing:
            if photo == ".DS_Store":
                continue
            try:
                img = Image.open(TRAINING_PATH + TRAINING_DATASET[label] + '/' + photo).convert('L')
                img_arr = np.array(img).ravel()
                if dedup:
                    if hash(str(img_arr)) in processed_image_hashes:
                        continue

                    processed_image_hashes.add(hash(str(img_arr)))
                images_and_labels.append([img_arr, label])
            except Exception as e:
                pass
    return images_and_labels


images_and_labels = add_pcs(dedup=False)
print len(images_and_labels)
print "=============================="
numpy.random.seed()


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

# ACC, TPR, TNR =[]

def train(j,k, epochs=50):
    d, l, size = random_split(images_and_labels)
    data_train = np.array(d[:int(size * 0.9)])
    label_train = np.array(l[:int(size * 0.9)])
    data_test = np.array(d[int(size * 0.1):])
    label_test = np.array(l[int(size * 0.1):])
    scaler = get_scaler(data_train)
    save_scaler(scaler)
    data_train = scale_data(data_train, scaler)
    data_test = scale_data(data_test, scaler)
    model = Sequential()
    model.add(Dense(j, input_dim=INPUT_SIZE, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data_train, label_train, validation_split=0.02, epochs=epochs)
    predictions = model.predict(data_test)
    # round predictions
    rounded_predictions = [round(x[0]) for x in predictions]
    [ACC, TPR, TNR] = compute_TPR_TNR(label_test, rounded_predictions)

    return (model, ACC, TPR, TNR)


model, ACC, TPR, TNR = train(25, 10, epochs=15)

model.save(KERAS_MODEL_PATH + TRAINING_DATASET[1] + ".h5")

print 'Accuracy is: ' + str(ACC) + ' True Positive Rate: ' + str(
        TPR) + ' True Negative Rate: ' + str(TNR)  + " "+ str(_j) + " k: " +str(_k)

