import time
import pandas as pd
import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import adam
from keras import callbacks, Model

from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score


def run_exp_model(X, y, X_val, y_val, X_test, y_test, nruns = 10):
    
    """
    Parameters:
    X, y = training pandas dataframe
    X_val, y_val = validation pandas dataframe
    X_test, y_test = testing pandas dataframe
    nruns = number of time the models should rn
    
    return:
    Dataframe with Model Run, Accuracy, Precision, Recall, F1 Score, AUC Score
    """
    
    score_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"])

    for i in range(0, nruns):
        start = time.time()

        model = Sequential()
        model.add(LSTM(
                 units=100,
                 return_sequences=True,
                 input_shape=(seq_length, nb_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(
                  units=50,
                  return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer = "adam", metrics=["accuracy"])

        model.fit(X, y, epochs=100, batch_size=128, validation_data=(X_val, y_val), verbose=0, class_weight=class_wt,
              shuffle= True ,callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
                                                                  verbose=0, mode='auto')])

        probabilities = model.predict_proba(X_test)
        predictions = [1 if i > 0.5 else 0 for i in probabilities]

        acc = accuracy_score(y_test,predictions)
        pr = precision_score(y_test,predictions)
        rc = recall_score(y_test,predictions)
        f1 = f1_score(y_test,predictions)
        auc = roc_auc_score(y_test,predictions)

        score_df = score_df.append({'Model':i+1, 'Accuracy': acc, 'Precision': pr, 'Recall': rc,
                                    'F1 Score': f1, 'AUC Score': auc}, ignore_index=True)
        end = time.time()
        print("Run:", i+1, " and Runtime:", np.round(end - start, 3), " Seconds", sep = "")        
    
    return score_df
