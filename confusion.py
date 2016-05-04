import pandas as pd
import numpy as np
import dicts
import matplotlib.pyplot as plt
from forests import *

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, name=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(dicts.country))
    plt.xticks(tick_marks, dicts.country.values(), rotation=45)
    plt.yticks(tick_marks, dicts.country.values())
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("{}_confusion_matrix.svg".format(name), format="svg")

pred_nodates = pd.read_csv("pred_train_users_nodates.csv")

trainx, trainy, test = transform_data(train_path="data_train_users_2.csv",
									  test_path="data_test_users.csv",xcols=dicts.default_cols,
									  parse_dates=[1,3],final_cols=dicts.columns)

true_y = np.array(trainy)

pred_y = np.array(pred_nodates.country1.map(dicts.country_ord))
cm = confusion_matrix(true_y, pred_y)

# log transform the matrix!
x = np.ma.log(cm)

plot_confusion_matrix(x.filled(0), name="no_dates")