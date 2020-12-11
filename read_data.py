# utilities
import re
import os
import pandas as pd
import time
import pickle
import scipy as sp
import numpy as np
# plotting
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim


def create_data_for_models(redo_preprocessing=True, fit_BoW=True, fit_TFIDF=True, fit_better_TFIDF=True,
                           create_word_vector_model=False, preprocessing_params=None):
    if preprocessing_params is None:
        preprocessing_params = [False, False]
    dataset, text, sentiment = read_dataset()
    if redo_preprocessing:
        preprocessing_step(text, preprocessing_params)
    X_train, X_test, y_train, y_test, unsplit_data = create_model_data(sentiment)
    save_label_data(y_train, y_test)
    t = time.time()
    if fit_BoW:
        fit_BoW_vectorizer(X_train, X_test)
    if fit_TFIDF:
        fit_TFIDF_vectorizer(X_train, X_test)
    if fit_better_TFIDF:
        fit_TFIDF_ngram_vectorizer(X_train, X_test)
    if create_word_vector_model:
        fit_word_vector_model(X_train, X_test, unsplit_data)
    print(f'Dataset processing complete. Time Taken: {round(time.time() - t)} seconds')

    # Plotting the distribution for dataset.
    make_graphs(dataset, X_train, X_test)
    # text analysis
    get_wordcloud()


def read_dataset():
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"

    dataset_filename = os.listdir("../project/input")[0]
    dataset_path = os.path.join("../project", "input", dataset_filename)
    print("Open file:", dataset_path)
    dataset = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, nrows=299)

    # Removing the unnecessary columns.
    dataset = dataset[['sentiment', 'text']]
    # Replacing the values to ease understanding.
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
    # Storing data in lists.
    text, sentiment = list(dataset['text']), list(dataset['sentiment'])
    return dataset, text, sentiment


def preprocessing_step(text, preprocessing_params):
    """
    Some of this is inspired from Nikit Periwal at
    https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :return:
    """
    t = time.time()
    print("Begin text preprocessing")
    processedtext, lexicon_analysis, polarity_shift_word = preprocess(text, preprocessing_params)
    print(f'Text Preprocessing complete. Time Taken: {round(time.time() - t)} seconds')
    # save the models for later use
    file = open('processedtext.pickle', 'wb')
    file2 = open('polarity_shift_word.pickle', 'wb')
    file3 = open('lexicon_analysis.pickle', 'wb')
    pickle.dump(processedtext, file)
    file.close()
    pickle.dump(polarity_shift_word, file2)
    file2.close()
    pickle.dump(lexicon_analysis, file3)
    file2.close()


def create_model_data(sentiment):
    TRAIN_SIZE = 0.8
    # Load
    file = open('processedtext.pickle', 'rb')
    processedtext = pickle.load(file)
    file.close()
    file = open('polarity_shift_word.pickle', 'rb')
    polarity_shift_word = pickle.load(file)
    file.close()
    file = open('lexicon_analysis.pickle', 'rb')
    lexicon_analysis = pickle.load(file)
    file.close()
    print("Splitting data")
    data = {'word_vector': processedtext,
            'lexicon_analysis': lexicon_analysis,
            'polarity_shift_word': polarity_shift_word,
            }
    df = pd.DataFrame(data, columns=['word_vector', 'lexicon_analysis', 'polarity_shift_word'])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df, sentiment, test_size=1 - TRAIN_SIZE, random_state=0)

    print("TRAIN size:", len(X_train))
    print("TEST size:", len(X_test))
    return X_train, X_test, y_train, y_test, df


def fit_BoW_vectorizer(X_train, X_test):
    t = time.time()
    tweet_vectoriser_Bow = CountVectorizer(ngram_range=(1, 1))
    X_train_BoW = vectorize(X_train, tweet_vectoriser_Bow)
    print(f'Bag of Words Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_Bow.get_feature_names()) + 2)
    X_test_BoW = transform(X_test, tweet_vectoriser_Bow)
    save_models(X_train_BoW, X_test_BoW, 'BoW')
    file = open('BoW_vectoriser.pickle', 'wb')
    pickle.dump(tweet_vectoriser_Bow, file)
    file.close()
    print(f'Bag of Words vectoriser fitted. Time Taken: {round(time.time() - t)} seconds')


def fit_TFIDF_vectorizer(X_train, X_test):
    t = time.time()
    tweet_vectoriser_TFIDF_no_ngram = TfidfVectorizer(ngram_range=(1, 1))
    X_train_TFIDF_no_ngram = vectorize(X_train, tweet_vectoriser_TFIDF_no_ngram)
    print(f'TFIDF Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_TFIDF_no_ngram.get_feature_names()) + 2)
    X_test_TFIDF_no_ngram = transform(X_test, tweet_vectoriser_TFIDF_no_ngram)
    save_models(X_train_TFIDF_no_ngram, X_test_TFIDF_no_ngram, 'TFIDF_no_ngram')
    file = open('TFIDF_no_ngram_vectoriser.pickle', 'wb')
    pickle.dump(tweet_vectoriser_TFIDF_no_ngram, file)
    file.close()
    print(f'TFIDF vectoriser fitted. Time Taken: {round(time.time() - t)} seconds')


def fit_TFIDF_ngram_vectorizer(X_train, X_test):
    t = time.time()
    tweet_vectoriser_TFIDF_with_ngram = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    X_train_TFIDF_with_ngram = vectorize(X_train, tweet_vectoriser_TFIDF_with_ngram)
    print(f'TFIDF_ngrams Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_TFIDF_with_ngram.get_feature_names()) + 2)
    X_test_with_ngram = transform(X_test, tweet_vectoriser_TFIDF_with_ngram)
    save_models(X_train_TFIDF_with_ngram, X_test_with_ngram, 'TFIDF_with_ngram')
    # save the models for later use
    file = open('TFIDF_ngram_vectoriser.pickle', 'wb')
    pickle.dump(tweet_vectoriser_TFIDF_with_ngram, file)
    file.close()
    print(f'2-ngram TFIDF vectoriser fitted. Time Taken: {round(time.time() - t)} seconds')


def fit_word_vector_model(X_train, X_test, df):
    t = time.time()
    # Load
    file = open('processedtext.pickle', 'rb')
    processedtext = pickle.load(file)
    file.close()

    # now word embedded vectors
    print("Now training word vector model")
    new_model, word2vec_model = create_word2vec(df['word_vector'])
    wordvec_arrays = np.zeros((len(processedtext), 202))
    for i in range(len(processedtext)):
        wordvec_arrays[i, :] = create_tweet_vector(processedtext[i].split(), 202, new_model)
    wordvec_df = pd.DataFrame(wordvec_arrays)

    wordvec_df_train = wordvec_df.loc[X_train.index]
    wordvec_df_test = wordvec_df.loc[X_test.index]

    save_models(wordvec_df_train, wordvec_df_test, 'word_vec')
    print(f'Word vector model fitted. Time Taken: {round(time.time() - t)} seconds')


def save_label_data(y_train, y_test):
    file = open('y_train.pickle', 'wb')
    pickle.dump(y_train, file)
    file.close()

    file = open('y_test.pickle', 'wb')
    pickle.dump(y_test, file)
    file.close()


def save_models(train, test, name):
    # save the models for later use
    train_name = 'X_train_' + name + '.pickle'
    test_name = 'X_test_' + name + '.pickle'
    file = open(train_name, 'wb')
    pickle.dump(train, file)
    file.close()
    file = open(test_name, 'wb')
    pickle.dump(test, file)
    file.close()


def vectorize(data, text_vectoriser):
    vectorized_data = sp.sparse.hstack((text_vectoriser.fit_transform(data['word_vector']),
                                        data[['lexicon_analysis', 'polarity_shift_word']].values),
                                       format='csr')
    return vectorized_data


def transform(data, text_vectoriser):
    vectorized_data = sp.sparse.hstack((text_vectoriser.transform(data['word_vector']),
                                        data[['lexicon_analysis', 'polarity_shift_word']].values),
                                       format='csr')
    return vectorized_data


def create_word2vec(preprocessed):
    """Part of this code was written by Nitin Garg at:
    https://www.kaggle.com/nitin194/twitter-sentiment-analysis-word2vec-doc2vec

    Training our own word vectors"""
    tokenized_tweet = preprocessed.apply(lambda x: x.split())  # tokenizing
    analyser = SentimentIntensityAnalyzer()

    model_w2v = gensim.models.Word2Vec(
        size=200,  # desired no. of features/independent variables
        window=5,  # context window size
        min_count=2,  # Ignores all words with total frequency lower than 2.
        sg=1,  # 1 for skip-gram model
        hs=0,
        negative=10,  # for negative sampling
        workers=32,  # no.of cores
        seed=34,
        iter=10  # no. of epochs
    )
    model_w2v.build_vocab(tokenized_tweet, progress_per=10000)

    model_w2v.train(tokenized_tweet, total_examples=len(preprocessed), epochs=20)
    # code to see dataframe
    ordered_vocab = [(term, voc.index, voc.count) for term, voc in model_w2v.wv.vocab.items()]
    ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])
    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    word_vectors = pd.DataFrame(model_w2v.wv.syn0[term_indices, :], index=ordered_terms)
    scores = []
    pol_shift_list = []
    for word in word_vectors.index:
        score = analyser.polarity_scores(word)
        scores.append(score['compound'])
        # see if its not or but
        pol_shift = 0
        if word == 'not' or word == 'but':
            pol_shift = 1
        pol_shift_list.append(pol_shift)
    word_vectors['scores'] = scores
    word_vectors['pol_shift'] = pol_shift_list

    return word_vectors, model_w2v


def create_tweet_vector(tokens, size, model_w2v):
    """
    Part of this code was written by Nitin Garg at:
    https://www.kaggle.com/nitin194/twitter-sentiment-analysis-word2vec-doc2vec
    "Since our data contains tweets and not just words, we’ll have to figure out a way to use the word vectors from
    word2vec model to create vector representation for an entire tweet. There is a simple solution to this problem,
    we can simply take mean of all the word vectors present in the tweet. The length of the resultant vector will be
    the same, i.e. 200. We will repeat the same process for all the tweets in our data and obtain their vectors.
    Now we have 200 word2vec features for our data.

    We will use the below function to create a vector for each tweet by taking the average of the vectors of the
    words present in the tweet."
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.loc[word].values.reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


def preprocess(textdata, preprocessing_params):
    """
    Some of the code in this function was written by Nikit Periwal at:
    https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data

    The Preprocessing steps taken are:

    Lower Casing: Each text is converted to lowercase.
    Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
    Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning.
    (eg: ":)" to "smile")
    Removing Usernames: Replace @Usernames with nothing.
    Removing Numbers: All numbers are replaced with
    Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
    Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
    Separating numbers and letters, replace all number with the word "number" (eg: "9am" to "number am")
    Removing Short Words: Words with length less than 2 are removed.
    Removing Stopwords: Stopwords are the English words which does not add much meaning to a sentence. They can safely
    be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
    Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)

    """

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'smile', ';D': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sad smile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    processedText = []
    part_of_speech = []
    lexicon_analysis = []
    polarity_shift_word = []

    # Create sentiment instensity analyser
    analyser = SentimentIntensityAnalyzer()
    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    maxAlphaPattern = "[^a-zA-Z0-9!?.]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    splitDigitChar = r"([0-9]+(\.[0-9]+)?)"
    i = 0
    for tweet in textdata:
        if i % 100 == 0:
            print("tweet #" + str(i))
        score = analyser.polarity_scores(tweet)
        lexicon_analysis.append(score['compound'])
        tweet = tweet.lower()
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "emoji_" + emojis[emoji])  # "EMOJI" + emojis[emoji])
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Remove @USERNAME.
        tweet = re.sub(userPattern, '', tweet)
        # Replace most non alphabets.
        tweet = re.sub(maxAlphaPattern, " ", tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        # separate numbers and letters, replace all number with the word "number"
        tweet = re.sub(splitDigitChar, r" number ", tweet)

        polarity = 0
        if 'not' in tweet or 'but' in tweet:
            polarity = 1

        # lemmatizing and stemming
        processed, pos = process_sentence(tweet.split(), preprocessing_params)
        processedText.append(processed)
        part_of_speech.append(pos)
        polarity_shift_word.append(polarity)
        i += 1
    return processedText, lexicon_analysis, polarity_shift_word


def process_sentence(tokens, preprocessing_params):
    if preprocessing_params[1]:
        stopwordlist = set(stopwords.words("english"))
    else:
        stopwordlist = []
    # Create Lemmatizer and Stemmer.
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_sentence = []
    partofspeech = []
    for word, tag in pos_tag(tokens):
        if len(word) > 1:
            if word not in stopwordlist:
                if tag.startswith('NN'):
                    pos = 'n'  # noun
                elif tag.startswith('VB'):
                    pos = 'v'  # verb
                elif tag.startswith('JJ'):
                    pos = 'a'  # adjective
                elif tag.startswith('RB'):
                    pos = 'r'  # adverb
                else:
                    pos = 'o'  # other

                if pos in ['n', 'v', 'a', 'r']:
                    word = lemmatizer.lemmatize(word, pos)
                else:
                    word = lemmatizer.lemmatize(word)
                # now stem
                if preprocessing_params[0]:
                    word = stemmer.stem(word)
                processed_sentence.append(word)
                partofspeech.append(pos)

    final_text = ' '.join(processed_sentence)
    final_pos = ' '.join(partofspeech)
    return final_text, final_pos


def get_wordcloud():
    # Load
    file = open('processedtext.pickle', 'rb')
    processedtext = pickle.load(file)
    file.close()
    data_neg = processedtext[:800000]
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800,
                   collocations=False).generate(" ".join(data_neg))
    plt.figure(figsize=(20, 20))
    plt.imshow(wc)
    plt.savefig('neg_wordcloud.png')
    data_pos = processedtext[800000:]
    wc = WordCloud(max_words=1000, width=1600, height=800,
                   collocations=False).generate(" ".join(data_pos))
    plt.figure(figsize=(20, 20))
    plt.imshow(wc)
    plt.savefig('pos_wordcloud.png')


def make_graphs(dataset, X_train, X_test):
    # Load
    file = open('processedtext.pickle', 'rb')
    processedtext = pickle.load(file)
    file.close()

    ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                                   legend=False)
    ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
    # Storing data in lists.
    fig = ax.get_figure()
    fig.savefig('bar_graph.png')
    pos_words = []
    neg_words = []
    for i in range(dataset.shape[0]):
        words = processedtext[i].split()
        if dataset["sentiment"][i] == 1:
            for word in words:
                pos_words.append(word.lower())
        else:
            for word in words:
                neg_words.append(word.lower())

    all_words = pos_words + neg_words
    plt.figure(figsize=(12, 5))
    plt.title('Top 25 most common words')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(all_words)
    fd.plot(25, cumulative=False)
    plt.savefig('common_words_all.png')

    plt.figure(figsize=(12, 5))
    plt.title('Top 25 most common words in positive tweets')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(pos_words)
    fd.plot(25, cumulative=False)
    plt.savefig('common_words_pos.png')

    plt.figure(figsize=(12, 5))
    plt.title('Top 25 most common words in negative tweets')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(neg_words)
    fd.plot(25, cumulative=False)
    plt.savefig('common_words_neg.png')
    plt.clf()  # clear figure
    plt.hist(X_train['word_vector'].str.len(), bins=20, label='train')
    plt.hist(X_test['word_vector'].str.len(), bins=20, label='test')
    plt.legend()
    plt.savefig('tweet_length.png')


if __name__ == '__main__':
    create_data_for_models(redo_preprocessing=True, fit_BoW=False, fit_TFIDF=False, fit_better_TFIDF=True,
                           create_word_vector_model=False)
