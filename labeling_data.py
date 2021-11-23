import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np
import tensorflow as tf
# np.set_printoptions(threshold=np.inf)
emo_lexci = pd.read_excel(r'DUTIR.xlsx', sheet_name='main')


emo_lab_ = list(emo_lexci['情感分类'])

for i in emo_lab_:
    if i == 'PA' or i == 'PE' or i == 'PD' or i == 'PH' or i == 'PG' or i == 'PB' or i == 'PK' or i == 'PC':
        emo_lab_[emo_lab_.index(i)] = 'zheng'
    elif i == 'NB' or i == 'NJ' or i == 'NH' or i == 'PF' or i == 'NI' or i == 'NC' or i == 'NG' or i == 'NE' or i == 'ND' or i == 'NN' or i == 'NK' or i == 'NL' or i == 'NAA':
        emo_lab_[emo_lab_.index(i)] = 'fu'

emo_distribution_ = dict([[i, emo_lab_.count(i)] for i in set(emo_lab_)])

labels = LabelBinarizer().fit_transform(emo_lab_)
labels = tf.squeeze(labels)
labels = tf.one_hot(labels, depth=2)
labels = np.array(labels)

with open(r'termWithLabsDL.pickle', 'wb')as f:
    pickle.dump(labels, f)
