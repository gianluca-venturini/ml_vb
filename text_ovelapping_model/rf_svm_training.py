import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from operator import itemgetter
import itertools
# from keras.models import Sequential
from scipy import misc
from PIL import Image
from PIL import ImageChops
import os
import glob
import random
# from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier

from text_ovelapping_model.params import model_path, size
from text_ovelapping_model.params import path

window_size = 30
step_seze = 20
sort = ['cropped-bb', 'cropped-gb']

ratio = 2
sample_size = [size, size*ratio]

# preparing photo windows from given photos
X = []
y = []
for i in range(2):
    listing = random.sample(os.listdir(path + sort[i]), sample_size[i])
    for photo in listing:
        if photo == ".DS_Store":
            continue
        try:
            img = Image.open(path + sort[i] + '/' + photo).convert('L')
            arr = np.array(img).ravel()
            X.append(arr)
            y.append(i)
        except:
            pass
data = pd.DataFrame({'X': X, 'y': y}).sample(frac=1).reset_index(drop=True)
X_train, X_val, X_test = np.split(data['X'], [int(len(X) * 0.6), int(len(X) * 0.8)])
y_train, y_val, y_test = np.split(data['y'], [int(len(X) * 0.6), int(len(X) * 0.8)])


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


results = list()

# Training SVM
# for i, j, t in itertools.product([0.01, 0.1, 1, 10, 100, 1000, 10000],
#                               [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],np.linspace(0.8,1.2,4)):
#     if (j==1000) and (t==0.8):
#         print i,
#     rbf_svc = svm.SVC(kernel='rbf', gamma=i, C=j, class_weight={0: ratio, 1: t}).fit(X_train.tolist(),
#                                                                                y_train.tolist())
#     lin_svc = svm.LinearSVC(C=j, class_weight={0: ratio, 1: t}).fit(X_train.tolist(),
#                                                                          y_train.tolist())
#     ACC_rbf, TPR_rbf, TNR_rbf = performance_measure(y_val.tolist(), rbf_svc.predict(X_val.tolist()))
#     ACC_lin, TPR_lin, TNR_lin = performance_measure(y_val.tolist(), lin_svc.predict(X_val.tolist()))
#     results.append(
#         ['rbf', rbf_svc, min(ACC_rbf, TPR_rbf, TNR_rbf), ACC_rbf, TPR_rbf, TNR_rbf, i, j,t])
#     results.append(
#         ['lin', lin_svc, min(ACC_lin, TPR_lin, TNR_lin), ACC_lin, TPR_lin, TNR_lin, i, j,t])

# Training random forest
for i in range(300, 302,2):
    print i,
    for j, k, t in itertools.product(range(1, 10), [5, 10, 20], np.linspace(0.8,1.2,4)):
        rf = RandomForestClassifier(n_estimators=i, max_depth=j, class_weight= {0: ratio, 1: t},
                                    random_state=0, min_samples_leaf=k).fit(X_train.tolist(),
                                                                            y_train.tolist())
        ACC_rf, TPR_rf, TNR_rf = performance_measure(y_val.tolist(), rf.predict(X_val.tolist()))
        results.append(['rf', rf, min(ACC_rf, TPR_rf, TNR_rf), ACC_rf, TPR_rf, TNR_rf, i, j, k, t])

# Choosing the best model
l = sorted(results, key=itemgetter(2), reverse=True)[0]
model = l[1]
print sorted(results, key=itemgetter(2), reverse=True)[:10]

with open(model_path, "wb") as f:
    pickle.dump(model,f)

ACC, TPR, TNR = performance_measure(y_test.tolist(), model.predict(X_test.tolist()))
print 'Model chosen is: ' + l[0]
print 'Accuracy is: ' + str(ACC) + ' True Positive Rate: ' + str(
    TPR) + ' True Negative Rate: ' + str(TNR) + ' ' + str(l[-3:])

