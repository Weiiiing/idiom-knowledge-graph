import os
import codecs
from keras_bert import load_trained_model_from_checkpoint
import numpy as np
from keras_bert import Tokenizer
import time
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Bidirectional, LSTM, GRU, RNN
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold
import logging
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from keras.layers import GlobalAveragePooling1D
# np.set_printoptions(threshold=np.inf)

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def label_trans(labels):
    new_labels = LabelBinarizer().fit_transform(labels)
    return new_labels

feature = read_pickle(r'vector_avg_all.pickle')
termWithLabs_labs = read_pickle(r'termWithLabsDL.pickle')
termsWithLabs_vec = []
for i in feature:
    for j in i:
        termsWithLabs_vec.append(j)
termsWithLabs_vec = np.array([[i] for i in termsWithLabs_vec])
print(termsWithLabs_vec.shape)
termWithLabs_labs_ = termWithLabs_labs.argmax(axis=1)


early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=2)
def bilstm_att(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(768,), return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.fit(train_X, train_Y, epochs=100, validation_data=(test_X, test_Y),
              batch_size=128, verbose=2, callbacks=[])

    model.save(r'model_BiLSTM_Attention_avg_meansyn.h5')
    y_pre = model.predict(test_X, batch_size=128)

    print(classification_report(test_Y.argmax(axis= 1), y_pre.argmax(axis= 1), digits= 5))
    report = classification_report(test_Y.argmax(axis=1), y_pre.argmax(axis=1), digits=5, output_dict=True)
    return report

skf = StratifiedKFold(n_splits=5)

label_0_p = []
label_0_r = []
label_0_f = []

label_1_p = []
label_1_r = []
label_1_f = []

acc = []
macro_p = []
macro_r = []
macro_f = []

train_idx = [0, 1, 2, 3] #根据迁移学习数据自行设置索引
test_idx = [4, 5] #根据迁移学习数据自行设置索引
train_X, train_Y = termsWithLabs_vec[train_idx], termWithLabs_labs_[train_idx]
test_X, test_Y = termsWithLabs_vec[test_idx], termWithLabs_labs_[test_idx]
print(train_X.shape)
print(test_X.shape)

train_X = train_X.astype('float64')
test_X = test_X.astype('float64')
train_Y = label_trans(train_Y)
test_Y = label_trans(test_Y)

train_Y = tf.squeeze(train_Y)
test_Y = tf.squeeze(test_Y)
train_Y = tf.one_hot(train_Y, depth=2)
test_Y = tf.one_hot(test_Y, depth=2)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

report = bilstm_att(train_X, train_Y, test_X, test_Y)
acc.append(report['accuracy'])
macro_p.append(report['macro avg']['precision'])
macro_r.append(report['macro avg']['recall'])
macro_f.append(report['macro avg']['f1-score'])

label_0_p.append(report['0']['precision'])
label_0_r.append(report['0']['recall'])
label_0_f.append(report['0']['f1-score'])

label_1_p.append(report['1']['precision'])
label_1_r.append(report['1']['recall'])
label_1_f.append(report['1']['f1-score'])


print('acc:' + str(sum(acc) ))
print('macro_p:' + str(sum(macro_p) ))
print('macro_r:' + str(sum(macro_r) ))
print('macro_f:' + str(sum(macro_f) ))

print('label_0_p:' + str(sum(label_0_p) ) + '\t' + 'label_0_r:' + str(sum(label_0_r) ) + '\t' + 'label_0_f:' + str(sum(label_0_f) ))
print('label_1_p:' + str(sum(label_1_p) ) + '\t' + 'label_1_r:' + str(sum(label_1_r) ) + '\t' + 'label_1_f:' + str(sum(label_1_f) ))
