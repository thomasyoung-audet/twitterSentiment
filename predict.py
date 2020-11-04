# utilities
import pickle
import pandas as pd
import read_data
import evaluate


def load_models():
    # Load the Naive Bayes model.
    file = open('Sentiment-BNB.pickle', 'rb')
    BNBmodel = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return BNBmodel, LRmodel


def load_data():
    # Load the vectoriser.
    file = open('vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load X_train
    file = open('X_train.pickle', 'rb')
    X_train = pickle.load(file)
    file.close()
    # Load X_train
    file = open('X_test.pickle', 'rb')
    X_test = pickle.load(file)
    file.close()
    # Load X_train
    file = open('y_train.pickle', 'rb')
    y_train = pickle.load(file)
    file.close()
    # Load X_train
    file = open('y_test.pickle', 'rb')
    y_test = pickle.load(file)
    file.close()

    return X_train, y_train, X_test, y_test, vectoriser


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(read_data.preprocess(text))
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectoriser = load_data()
    evaluate.create_models(X_train, y_train, X_test, y_test, vectoriser)
    # Loading the models.
    BNBmodel, LRmodel = load_models()

    # Text to classify should be in a list.
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    df = predict(vectoriser, LRmodel, text)
    print(df.head())
    df = predict(vectoriser, BNBmodel, text)
    print(df.head())
