# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import pandas as pd
from scipy import io as sio

# helpers
import time

# classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
# split data set

# import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GRU, Dropout, LSTM, Bidirectional, SimpleRNN, Input, Concatenate, Conv1D, Flatten, MaxPooling1D, Reshape
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBClassifier

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

execution = {}
accuracies = {}
precisions = {}
recalls = {}

# num_classes = 6
epochs_n = 100
units = 100

def evaluate(y_test, y_pred, modelName='GRU', wordemb='w2v_sg', iters=0):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)
        try:
            # if an error appears here
            # get a random class
            elem[np.isnan(elem)] = 0
            line[elem.tolist().index(max(elem.tolist()))] = 1
        except:
            print("Error for getting predicted class")
            print(elem.tolist())
            line[rnd.randint(0, len(elem)-1)] = 1
        y_pred_norm.append(line)

    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)
    accuracy = accuracy_score(y_t, y_p)
    if num_classes == 2:
        precision = precision_score(y_t, y_p, average='binary')
        recall = recall_score(y_t, y_p, average='binary')
    else:
        precision = precision_score(y_t, y_p, average='weighted')
        recall = recall_score(y_t, y_p, average='weighted')


    accuracies[wordemb][modelName].append(accuracy)
    precisions[wordemb][modelName].append(precision)
    recalls[wordemb][modelName].append(recall)

    print(modelName, wordemb, "Accuracy", accuracy)
    print(modelName, wordemb, "Precision", precision)
    print(modelName, wordemb, "Recall", recall)
    print("accuracies['", wordemb, "']['", modelName, "'].append(", accuracy, ")")
    print("precisions['", wordemb, "']['", modelName, "'].append(", precision, ")")
    print("recalls['", wordemb, "']['", modelName, "'].append(", recall, ")")
    print(modelName, wordemb, "Report", classification_report(y_t, y_p))
    print(modelName, wordemb, confusion_matrix(y_t, y_p))
    return y_p, y_t

if __name__ =="__main__":
    dir_name = sys.argv[1] # the directory
    NUM_ITER = int(sys.argv[2])

    ##################### LABELS ############################

    y = sio.loadmat(os.path.join(dir_name, 'labels.mat'))['y'][0]
    print(y)
    print(len(y))

    num_classes = len(np.unique(y))
    print(num_classes)

    for wordemb in ['D2V_TFIDF', 'D2V_W2V_CBOW', 'D2V_W2V_SG', 'D2V_FT_CBOW', 'D2V_FT_SG', 'D2V_GLOVE', 'D2V_BERT', 'D2V_RoBERTa', 'D2V_BART']:
        accuracies[wordemb] = {}
        precisions[wordemb] = {}
        recalls[wordemb] = {}
        execution[wordemb] = {}

        execution[wordemb]["Perceptron"] = []
        accuracies[wordemb]["Perceptron"] = []
        precisions[wordemb]["Perceptron"] = []
        recalls[wordemb]["Perceptron"] = []

        execution[wordemb]["MLP"] = []
        accuracies[wordemb]["MLP"] = []
        precisions[wordemb]["MLP"] = []
        recalls[wordemb]["MLP"] = []

        execution[wordemb]["LSTM"] = []
        accuracies[wordemb]["LSTM"] = []
        precisions[wordemb]["LSTM"] = []
        recalls[wordemb]["LSTM"] = []

        execution[wordemb]["BiLSTM"] = []
        accuracies[wordemb]["BiLSTM"] = []
        precisions[wordemb]["BiLSTM"] = []
        recalls[wordemb]["BiLSTM"] = []

        execution[wordemb]["GRU"] = []
        accuracies[wordemb]["GRU"] = []
        precisions[wordemb]["GRU"] = []
        recalls[wordemb]["GRU"] = []

        execution[wordemb]["BiGRU"] = []
        accuracies[wordemb]["BiGRU"] = []
        precisions[wordemb]["BiGRU"] = []
        recalls[wordemb]["BiGRU"] = []

        execution[wordemb]["NB"] = []
        accuracies[wordemb]["NB"] = []
        precisions[wordemb]["NB"] = []
        recalls[wordemb]["NB"] = []

        execution[wordemb]["XGB"] = []
        accuracies[wordemb]["XGB"] = []
        precisions[wordemb]["XGB"] = []
        recalls[wordemb]["XGB"] = []

        X_D2V = sio.loadmat(os.path.join(dir_name, wordemb + '.mat'))['X']
        print(X_D2V)
        print(np.amax(X_D2V))


        for idx in range(0, NUM_ITER):
            x_train, x_test, y_train, y_test = train_test_split(X_D2V, y, test_size=0.20, shuffle = True, stratify = y)
            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            x_vec_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
            x_vec_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
            y_vec_train = to_categorical(y_train, num_classes=num_classes)
            y_vec_test = to_categorical(y_test, num_classes=num_classes)

            print(x_vec_train.shape)
            print(x_vec_test.shape)
            print(y_vec_train.shape)
            print(y_vec_test.shape)

            # Perceptron
            start_time = time.time()
            input_layer = Input(shape=(x_train.shape[1]), name = 'Input')
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(input_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="Perceptron")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["Perceptron"].append(exc_time)
            print('Perceptron', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['Perceptron'].append(", exc_time, ")")

            # MLP
            start_time = time.time()
            input_layer = Input(shape=(x_train.shape[1]), name = 'Input')
            dense_layer = Dense(units, activation='sigmoid', name='Dense')(input_layer)
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(dense_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="MLP")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["MLP"].append(exc_time)
            print('MLP', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['MLP'].append(", exc_time, ")")

            # LSTM
            start_time = time.time()
            input_layer = Input(shape=(x_vec_train.shape[1], x_vec_train.shape[2]), name = 'Input')
            lstm_layer = LSTM(units = units, name='LSTM')(input_layer)
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(lstm_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="LSTM")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_vec_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_vec_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["LSTM"].append(exc_time)
            print('LSTM', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['LSTM'].append(", exc_time, ")")

            #  BiLSTM
            start_time = time.time()
            input_layer = Input(shape=(x_vec_train.shape[1], x_vec_train.shape[2]), name = 'Input')
            lstm_layer = Bidirectional(LSTM(units = units), name='BiLSTM')(input_layer)
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(lstm_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="BiLSTM")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_vec_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_vec_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["BiLSTM"].append(exc_time)
            print('BiLSTM', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['BiLSTM'].append(", exc_time, ")")


            # GRU
            start_time = time.time()
            input_layer = Input(shape=(x_vec_train.shape[1], x_vec_train.shape[2]), name = 'Input')
            lstm_layer = GRU(units = units, name='GRU')(input_layer)
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(lstm_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="GRU")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_vec_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_vec_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["GRU"].append(exc_time)
            print('GRU', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['GRU'].append(", exc_time, ")")

            #  BiGRU
            start_time = time.time()
            input_layer = Input(shape=(x_vec_train.shape[1], x_vec_train.shape[2]), name = 'Input')
            lstm_layer = Bidirectional(GRU(units = units), name='BiGRU')(input_layer)
            output_layer = Dense(num_classes, activation='sigmoid', name='Output')(lstm_layer)

            model = Model(inputs=input_layer, outputs=output_layer, name="BiGRU")

            if num_classes == 2:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

            history = model.fit(x = x_vec_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

            y_pred = model.predict(x_vec_test, verbose=False)
            evaluate(y_vec_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
            end_time = time.time()

            exc_time = (end_time - start_time)
            execution[wordemb]["BiGRU"].append(exc_time)
            print('BiGRU', wordemb, "Time", exc_time)
            print("execution['", wordemb, "']['BiGRU'].append(", exc_time, ")")

            start_time = time.time()
            try:
                print('MultinomialNB')
                nb = MultinomialNB().fit(x_train, y_train)
                y_pred = nb.predict(x_test)
            except:
                print('GaussianNB')
                nb = GaussianNB().fit(x_train, y_train)
                y_pred = nb.predict(x_test)

            end_time = time.time()
            exc_time = (end_time - start_time)

            accuracy = accuracy_score(y_test, y_pred)
            if num_classes == 2:
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            accuracies[wordemb]["NB"].append(accuracy)
            precisions[wordemb]["NB"].append(precision)
            recalls[wordemb]["NB"].append(recall)
            execution[wordemb]["NB"].append(exc_time)

            print('NB', wordemb, "Accuracy", accuracy)
            print('NB', wordemb, "Precision", precision)
            print('NB', wordemb, "Recall", recall)
            print('NB', wordemb, "Time", exc_time)
            print('NB', wordemb, "Report", classification_report(y_test, y_pred))
            print('NB', wordemb, confusion_matrix(y_test, y_pred))

            print("accuracies['", wordemb, "']['NB'].append(", accuracy, ")")
            print("precisions['", wordemb, "']['NB'].append(", precision, ")")
            print("recalls['", wordemb, "']['NB'].append(", recall, ")")
            print("execution['", wordemb, "']['NB'].append(", exc_time, ")")

            print('XGBoost')

            xgb = XGBClassifier(use_label_encoder=False)
            xgb.fit(x_train,y_train)
            y_pred = xgb.predict(x_test)

            end_time = time.time()
            exc_time = (end_time - start_time)

            accuracy = accuracy_score(y_test, y_pred)
            if num_classes == 2:
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            accuracies[wordemb]["XGB"].append(accuracy)
            precisions[wordemb]["XGB"].append(precision)
            recalls[wordemb]["XGB"].append(recall)
            execution[wordemb]["XGB"].append(exc_time)

            print('XGB', wordemb, "Accuracy", accuracy)
            print('XGB', wordemb, "Precision", precision)
            print('XGB', wordemb, "Recall", recall)
            print('XGB', wordemb, "Time", exc_time)
            print('XGB', wordemb, "Report", classification_report(y_test, y_pred))
            print('XGB', wordemb, confusion_matrix(y_test, y_pred))

            print("accuracies['", wordemb, "']['XGB'].append(", accuracy, ")")
            print("precisions['", wordemb, "']['XGB'].append(", precision, ")")
            print("recalls['", wordemb, "']['XGB'].append(", recall, ")")
            print("execution['", wordemb, "']['XGB'].append(", exc_time, ")")


    for wordemb in ['D2V_TFIDF', 'D2V_W2V_CBOW', 'D2V_W2V_SG', 'D2V_FT_CBOW', 'D2V_FT_SG', 'D2V_GLOVE', 'D2V_BERT', 'D2V_RoBERTa', 'D2V_BART']:
        print("\n\n========================================\n\n")

        print("NB", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["NB"])*100, 2), round(np.std(accuracies[wordemb]["NB"])*100, 2))
        print("NB", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["NB"])*100, 2), round(np.std(precisions[wordemb]["NB"])*100, 2))
        print("NB", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["NB"])*100, 2), round(np.std(recalls[wordemb]["NB"])*100, 2))
        print("NB", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["NB"]), 2), round(np.std(execution[wordemb]["NB"]), 2))

        print("XGB", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["XGB"])*100, 2), round(np.std(accuracies[wordemb]["XGB"])*100, 2))
        print("XGB", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["XGB"])*100, 2), round(np.std(precisions[wordemb]["XGB"])*100, 2))
        print("XGB", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["XGB"])*100, 2), round(np.std(recalls[wordemb]["XGB"])*100, 2))
        print("XGB", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["XGB"]), 2), round(np.std(execution[wordemb]["XGB"]), 2))

        print("Perceptron", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["Perceptron"])*100, 2), round(np.std(accuracies[wordemb]["Perceptron"])*100, 2))
        print("Perceptron", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["Perceptron"])*100, 2), round(np.std(precisions[wordemb]["Perceptron"])*100, 2))
        print("Perceptron", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["Perceptron"])*100, 2), round(np.std(recalls[wordemb]["Perceptron"])*100, 2))
        print("Perceptron", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["Perceptron"]), 2), round(np.std(execution[wordemb]["Perceptron"]), 2))
        
        print("MLP", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["MLP"])*100, 2), round(np.std(accuracies[wordemb]["MLP"])*100, 2))
        print("MLP", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["MLP"])*100, 2), round(np.std(precisions[wordemb]["MLP"])*100, 2))
        print("MLP", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["MLP"])*100, 2), round(np.std(recalls[wordemb]["MLP"])*100, 2))
        print("MLP", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["MLP"]), 2), round(np.std(execution[wordemb]["MLP"]), 2))

        print("LSTM", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["LSTM"])*100, 2), round(np.std(accuracies[wordemb]["LSTM"])*100, 2))
        print("LSTM", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["LSTM"])*100, 2), round(np.std(precisions[wordemb]["LSTM"])*100, 2))
        print("LSTM", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["LSTM"])*100, 2), round(np.std(recalls[wordemb]["LSTM"])*100, 2))
        print("LSTM", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["LSTM"]), 2), round(np.std(execution[wordemb]["LSTM"]), 2))

        print("BiLSTM", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiLSTM"])*100, 2), round(np.std(accuracies[wordemb]["BiLSTM"])*100, 2))
        print("BiLSTM", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["BiLSTM"])*100, 2), round(np.std(precisions[wordemb]["BiLSTM"])*100, 2))
        print("BiLSTM", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["BiLSTM"])*100, 2), round(np.std(recalls[wordemb]["BiLSTM"])*100, 2))
        print("BiLSTM", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiLSTM"]), 2), round(np.std(execution[wordemb]["BiLSTM"]), 2))

        print("GRU", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU"])*100, 2), round(np.std(accuracies[wordemb]["GRU"])*100, 2))
        print("GRU", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["GRU"])*100, 2), round(np.std(precisions[wordemb]["GRU"])*100, 2))
        print("GRU", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["GRU"])*100, 2), round(np.std(recalls[wordemb]["GRU"])*100, 2))
        print("GRU", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU"]), 2), round(np.std(execution[wordemb]["GRU"]), 2))

        print("BiGRU", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU"])*100, 2))
        print("BiGRU", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["BiGRU"])*100, 2), round(np.std(precisions[wordemb]["BiGRU"])*100, 2))
        print("BiGRU", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["BiGRU"])*100, 2), round(np.std(recalls[wordemb]["BiGRU"])*100, 2))
        print("BiGRU", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU"]), 2), round(np.std(execution[wordemb]["BiGRU"]), 2))
