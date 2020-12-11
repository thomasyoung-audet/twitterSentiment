# utilities
import pickle
import pandas as pd
import read_data
import evaluate


def fit_and_evaluate_models(BoW, TFIDF, ngramTFIDF, wordvec):
    if BoW:
        y_test, y_train, X_test, X_train, vectorizer = load_in_data(BoW)
        print("=========Running models on Bag of Words data===============")
        evaluate.create_models(X_train, y_train, X_test, y_test, "BoW")
    if TFIDF:
        y_test, y_train, X_test, X_train, vectorizer = load_in_data(False, TFIDF)
        print("=========Running models on TFIDF data===============")
        evaluate.create_models(X_train, y_train, X_test, y_test, "TFIDF_no_ngram")
    if ngramTFIDF:
        y_test, y_train, X_test, X_train, vectorizer = load_in_data(False, False, ngramTFIDF)
        print("=========Running models on 2-ngram TFIDF data===============")
        evaluate.create_models(X_train, y_train, X_test, y_test, "TFIDF_ngram")
    if wordvec:
        y_test, y_train, X_test, X_train, vectorizer = load_in_data(False, False, False, wordvec)
        print("=========Running models on Word Vector data===============")
        evaluate.create_models(X_train, y_train, X_test, y_test, "word_vec")


def load_in_data(BoW=False, TFIDF=False, ngramTFIDF=False, wordvec=False):
    # load in data
    y_test, y_train = load_Y()
    if BoW:
        X_test_BoW = load_test_data('BoW')
        X_train_BoW = load_train_data('BoW')
        tweet_vectoriser_Bow = load_vectorizers('BoW')
        return y_test, y_train, X_test_BoW, X_train_BoW, tweet_vectoriser_Bow
    if TFIDF:
        X_test_TFIDF_no_ngram = load_test_data('TFIDF_no_ngram')
        X_train_TFIDF_no_ngram = load_train_data('TFIDF_no_ngram')
        tweet_vectoriser_TFIDF_no_ngram = load_vectorizers('TFIDF_no_ngram')
        return y_test, y_train, X_test_TFIDF_no_ngram, X_train_TFIDF_no_ngram, tweet_vectoriser_TFIDF_no_ngram
    if ngramTFIDF:
        X_test_TFIDF_with_ngram = load_test_data('TFIDF_with_ngram')
        X_train_TFIDF_with_ngram = load_train_data('TFIDF_with_ngram')
        tweet_vectoriser_TFIDF_with_ngram = load_vectorizers('TFIDF_ngram')
        return y_test, y_train, X_test_TFIDF_with_ngram, X_train_TFIDF_with_ngram, tweet_vectoriser_TFIDF_with_ngram
    if wordvec:
        wordvec_df_test = load_test_data('word_vec')
        wordvec_df_train = load_train_data('word_vec')
        return y_test, y_train, wordvec_df_test, wordvec_df_train, False


def load_models(model, data_preprocess_type):
    file = open('Sentiment-' + model + '_' + data_preprocess_type + '.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    return model


def load_train_data(name):
    # Load X_train
    file = open('X_train_' + name + '.pickle', 'rb')
    X_train = pickle.load(file)
    file.close()
    return X_train


def load_test_data(name):
    # Load X_test
    file = open('X_test_' + name + '.pickle', 'rb')
    X_test = pickle.load(file)
    file.close()
    return X_test


def load_vectorizers(name):
    # Load the vectoriser.
    file = open(name + '_vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    return vectoriser


def load_Y():
    # Load the label data.
    file = open('y_test.pickle', 'rb')
    y_test = pickle.load(file)
    file.close()
    file = open('y_train.pickle', 'rb')
    y_train = pickle.load(file)
    file.close()

    return y_test, y_train


def predict(data_preprocess_type, model, text, preprocessing_params):
    # load in the model we want to use to predict the sentiment
    ml_model = load_models(model, data_preprocess_type)
    # load the method with which we preprocess the text
    tweet_vectoriser = load_vectorizers(data_preprocess_type)

    # preprocess the text to predict
    processedtext, lexicon_analysis, polarity_shift_word = read_data.preprocess(text, preprocessing_params)
    data = {'word_vector': processedtext,
            'lexicon_analysis': lexicon_analysis,
            'polarity_shift_word': polarity_shift_word,
            }

    df = pd.DataFrame(data, columns=['word_vector', 'lexicon_analysis', 'polarity_shift_word'])
    # Predict the sentiment
    textdata = read_data.transform(df, tweet_vectoriser)
    sentiment = ml_model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


def predict_text_sentiment(vectorizer, model, text, preprocessing_params):
    df = predict(vectorizer, model, text, preprocessing_params)
    print("=========" + model + " prediction =========")
    print(df.head())
