# utilities
import os
import pickle
import numpy as np
import time


# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def model_evaluate(model, name, X_test, y_test):
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
                    xticklabels=categories, yticklabels=categories, annot_kws={"fontsize": 8})

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title(name + " Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.savefig(name + " Confusion Matrix.png")
    plt.clf()


def runBernouliNB_Model(X_train, y_train):
    t = time.time()
    BNBmodel = BernoulliNB(alpha=2)
    BNBmodel.fit(X_train, y_train)
    print(f'Bernoulli Naive Bayes fit complete. Time Taken: {round(time.time() - t)} seconds')
    return BNBmodel


def runLinearSVC_Model(X_train, y_train):
    t = time.time()
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    print(f'Linear SVC fit complete. Time Taken: {round(time.time() - t)} seconds')
    return SVCmodel


def runLogReg_Model(X_train, y_train):
    t = time.time()
    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    print(f'Logistic Regression fit complete. Time Taken: {round(time.time() - t)} seconds')
    return LRmodel


def create_models(X_train, y_train, X_test, y_test, data_name):
    print("Naive Bayes")
    BNBmodel = runBernouliNB_Model(X_train, y_train)
    model_evaluate(BNBmodel, "Bernouli", X_test, y_test)
    print("SVC")
    SVCmodel = runLinearSVC_Model(X_train, y_train)
    model_evaluate(SVCmodel, "SVC", X_test, y_test)
    print("Logistic Regression")
    LRmodel = runLogReg_Model(X_train, y_train)
    model_evaluate(LRmodel, "Log Reg", X_test, y_test)

    # save the models for later use
    file = open('Sentiment-LR_' + data_name + '.pickle', 'wb')
    pickle.dump(LRmodel, file)
    file.close()

    file = open('Sentiment-BNB_' + data_name + '.pickle', 'wb')
    pickle.dump(BNBmodel, file)
    file.close()

    file = open('Sentiment-SVC_' + data_name + '.pickle', 'wb')
    pickle.dump(SVCmodel, file)
    file.close()
