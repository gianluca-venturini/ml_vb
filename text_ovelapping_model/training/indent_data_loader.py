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

path = '/Users/ran/Documents/ML_bad_lf/indent/'
scaler_path = path+"scaler"
keras_model_path = path+"model"

sort = ['indent', 'indent-bad']

sample_size = [6, 9]
input = 1500


def preprocess_data(X_user_train,  X_test):
    print "0000000000"
    print X_user_train
    print "0000000000"
    scaler = StandardScaler().fit(X_user_train )
    with open(scaler_path+ sort[1], "wb") as f:
        pickle.dump(scaler, f)

    return  scaler.transform(X_user_train).tolist(),  scaler.transform(X_test).tolist()


# preparing photo windows from given photos
def add_pcs(dedup=True):
    a = []
    s = set()
    for i in [0, 1]:
        print sample_size[i]
        print len(os.listdir(path + sort[i]))
        print sort[i]
        listing = random.sample(os.listdir(path + sort[i]), sample_size[i])
        for photo in listing:
            try:
                # img = Image.open(path + sort[i] + '/' + photo).convert('L')
                # arr = np.array(img).ravel()
                with open(path + sort[i] + '/' + photo, "rb") as f:
                    arr_tup = pickle.load(f)
                arr = []
                for c, b in arr_tup:
                    # arr.append(c)
                    arr.append(b)
                print "odododododo"
                print (1200 - len(arr))
                print "odododododo"
                zeros = [0] * (1500 - len(arr))
                arr.extend(zeros)

                if len(arr)==0:
                    continue
                # arr = np.array(arr)
                if dedup:
                    if hash(str(arr)) in s:
                        continue
                    s.add(hash(str(arr)))
                a.append([arr, i])
            except Exception as e:
                # raise e
                print e
    return a


x_y = add_pcs(dedup=False)
print len(x_y)
print "=============================="
numpy.random.seed()


def random_split():
    global size, x, d, l
    random.shuffle(x_y)
    size = len(x_y)
    d = [x[0] for x in x_y]
    l = [x[1] for x in x_y]



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
    global model, x, ACC, TPR, TNR
    random_split()
    # data = np.array(d[:int(size * 0.9)])
    # label = np.array(l[:int(size * 0.9)])
    # d_v = np.array(d[int(size * 0.1):])
    # l_v = np.array(l[int(size * 0.1):])
    #
    data = np.array(d)
    label = np.array(l)

    # data, d_v = preprocess_data(data, label)
    data = np.array(d)
    model = Sequential()
    model.add(Dense(j, input_dim=input, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(k, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, label, validation_split=0.2, epochs=epochs)
    predictions = model.predict(d_v)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    [ACC, TPR, TNR] = compute_TPR_TNR(l_v, rounded)


train(25,10,epochs=1000)

model.save(keras_model_path+sort[1]+".h5")

print 'Accuracy is: ' + str(ACC) + ' True Positive Rate: ' + str(
        TPR) + ' True Negative Rate: ' + str(TNR)  + " "+ str(_j) + " k: " +str(_k)

