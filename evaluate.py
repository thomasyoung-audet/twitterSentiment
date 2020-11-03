# utilities
import os
import pickle
import numpy as np
import pandas as pd
import time
import read_data

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


def model_evaluate(model, name):
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
    plt.savefig(name + " Confusion Matrix.png")
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


def create_models(X_train, y_train):
    BNBmodel = runBernouliNB_Model(X_train, y_train)
    SVCmodel = runLinearSVC_Model(X_train, y_train)
    LRmodel = runLogReg_Model(X_train, y_train)
    model_evaluate(BNBmodel, "Bernouli")
    model_evaluate(SVCmodel, "SVC")
    vectoriser = model_evaluate(LRmodel, "Log Reg")
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
