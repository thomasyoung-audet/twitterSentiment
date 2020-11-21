# utilities
import pickle
import pandas as pd
import read_data
import evaluate


def load_models(name):
    # Load the Naive Bayes model.
    file = open('Sentiment-BNB_' + name + '.pickle', 'rb')
    BNBmodel = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-LR_' + name + '.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return BNBmodel, LRmodel


def load_train_data():
    # Load X_train
    file = open('X_train_BoW.pickle', 'rb')
    X_train_BoW = pickle.load(file)
    file.close()
    # Load X_train
    file = open('X_train_TFIDF_no_ngram.pickle', 'rb')
    X_train_TFIDF_no_ngram = pickle.load(file)
    file.close()
    # Load X_train
    file = open('X_train_TFIDF_with_ngram.pickle', 'rb')
    X_train_TFIDF_with_ngram = pickle.load(file)
    file.close()

    return X_train_BoW, X_train_TFIDF_no_ngram, X_train_TFIDF_with_ngram


def load_test_data():
    # Load X_test
    file = open('X_test_BoW.pickle', 'rb')
    X_test_BoW = pickle.load(file)
    file.close()
    # Load X_test
    file = open('X_test_TFIDF_no_ngram.pickle', 'rb')
    X_test_TFIDF_no_ngram = pickle.load(file)
    file.close()
    # Load X_test
    file = open('X_test_TFIDF_with_ngram.pickle', 'rb')
    X_test_TFIDF_with_ngram = pickle.load(file)
    file.close()

    return X_test_BoW, X_test_TFIDF_no_ngram, X_test_TFIDF_with_ngram


def load_vectorizers():
    # Load the vectoriser.
    file = open('vectoriser.pickle', 'rb')
    vectorisers = pickle.load(file)
    file.close()

    return vectorisers[0], vectorisers[1], vectorisers[2]


def load_Y():
    # Load the vectoriser.
    file = open('y_test.pickle', 'rb')
    y_test = pickle.load(file)
    file.close()
    file = open('y_train.pickle', 'rb')
    y_train = pickle.load(file)
    file.close()

    return y_test, y_train


def predict(tweet_vectoriser, model, text):
    # Predict the sentiment
    part_of_speech, processedtext, lexicon_analysis, polarity_shift_word, longest = read_data.preprocess(text)
    data = {'part_of_speech': part_of_speech,
            'word_vector': processedtext,
            'lexicon_analysis': lexicon_analysis,
            'polarity_shift_word': polarity_shift_word,
            }

    df = pd.DataFrame(data, columns=['part_of_speech', 'word_vector', 'lexicon_analysis', 'polarity_shift_word'])
    textdata = read_data.transform(df, tweet_vectoriser)
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
    y_test, y_train = load_Y()
    X_test_BoW, X_test_TFIDF_no_ngram, X_test_TFIDF_with_ngram = load_test_data()
    X_train_BoW, X_train_TFIDF_no_ngram, X_train_TFIDF_with_ngram = load_train_data()
    tweet_vectoriser_Bow, tweet_vectoriser_TFIDF_no_ngram, tweet_vectoriser_TFIDF_with_ngram = load_vectorizers()
    print("Running models on Bag of Words data")
    evaluate.create_models(X_train_BoW, y_train, X_test_BoW, y_test, "Bow")
    # Loading the models.
    BNBmodel_BoW, LRmodel_BoW = load_models("BoW")
    print("Running models on TFIDF data")
    evaluate.create_models(X_train_TFIDF_no_ngram, y_train, X_test_TFIDF_no_ngram, y_test, "TFIDF_no_ngrams")
    # Loading the models.
    BNBmodel_TFIDF_no, LRmodel_TFIDF_no = load_models("TFIDF_no_ngrams")
    print("Running models on TFIDF data with ngrams")
    evaluate.create_models(X_train_TFIDF_with_ngram, y_train, X_test_TFIDF_with_ngram, y_test, "TFIDF_ngrams")
    # Loading the models.
    BNBmodel_TFIDF_ngram, LRmodel_TFIDF_ngram = load_models("TFIDF_ngrams")

    # Text to classify should be in a list.
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    df = predict(tweet_vectoriser_Bow, LRmodel_BoW, text)
    print("=========LR model=========")
    print(df.head())
    df = predict(tweet_vectoriser_Bow, BNBmodel_BoW, text)
    print("=========BNB model========")
    print(df.head())
