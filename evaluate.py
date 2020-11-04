# utilities
import os
import pickle
import numpy as np
import read_data

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# keras
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
# from keras.layers import SimpleRNN, Dense, Activation, Embedding, LSTM, SpatialDropout1D


def scikit_model_evaluate(model, name):
    X_train, X_test, y_train, y_test, vectorizer = read_data.read()
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title(name + " Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.savefig(name + " Confusion Matrix.png", dpi=100)
    return vectorizer


def runBernouliNB_Model(X_train, y_train):
    BNBmodel = BernoulliNB(alpha=2)
    BNBmodel.fit(X_train, y_train)
    return BNBmodel


def runLinearSVC_Model(X_train, y_train):
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    return SVCmodel


def runLogReg_Model(X_train, y_train):
    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    return LRmodel


# def runRNN_Model(X_train, y_train, X_test, y_test):
#     rnn = Sequential()
#     rnn.add(Embedding(num_words, 16, input_length=(maxlen)))  # 32
#     rnn.add(SimpleRNN(16, input_shape=(num_words, maxlen), return_sequences=False, activation="tanh"))
#     rnn.add(Dense(1))
#     rnn.add(Activation("sigmoid"))
#
#     print(rnn.summary())
#     rnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # rmsprop
#     history = rnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1)

def runLSTM_Model(X_train, y_train):
    embed_dim = 128
    lstm_out = 196
    max_fatures = 2000
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    batch_size = 32
    model.fit(X_train, y_train, epochs=7, batch_size=batch_size, verbose=2)  # up epoch number
    return model


def create_models(X_train, y_train):
    BNBmodel = runBernouliNB_Model(X_train, y_train)
    scikit_model_evaluate(BNBmodel, "Bernouli")
    SVCmodel = runLinearSVC_Model(X_train, y_train)
    scikit_model_evaluate(SVCmodel, "SVC")
    LRmodel = runLogReg_Model(X_train, y_train)
    vectoriser = scikit_model_evaluate(LRmodel, "Log Reg")
    # save the models for later use
    file = open('vectoriser-ngram-(1,2).pickle', 'wb')
    pickle.dump(vectoriser, file)
    file.close()

    file = open('Sentiment-LR.pickle', 'wb')
    pickle.dump(LRmodel, file)
    file.close()

    file = open('Sentiment-BNB.pickle', 'wb')
    pickle.dump(BNBmodel, file)
    file.close()
