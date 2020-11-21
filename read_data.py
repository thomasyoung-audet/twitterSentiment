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
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import sentiwordnet as swn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim


def read():
    """
    Some of this is inspired from Nikit Periwal at
    https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :return:
    """
    t = time.time()
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    TRAIN_SIZE = 0.8

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

    part_of_speech, processedtext, lexicon_analysis, polarity_shift_word, longest = preprocess(text)
    # padding
    # for i in range(len(lexicon_analysis)):
    #     padding_len = longest - len(lexicon_analysis[i])
    #     lexicon_analysis[i] = lexicon_analysis[i] + [0.0] * padding_len
    # # save the models for later use
    # file = open('processedtext.pickle', 'wb')
    # file2 = open('polarity_scores.pickle', 'wb')
    # pickle.dump(processedtext, file)
    # file.close()
    # pickle.dump(polarity_scores, file2)
    # file2.close()
    # Load
    # file = open('processedtext.pickle', 'rb')
    # processedtext = pickle.load(file)
    # file.close()
    # file = open('polarity_scores.pickle', 'rb')
    # polarity_scores = pickle.load(file)
    # file.close()
    print(f'Text Preprocessing complete.')

    data = {'part_of_speech': part_of_speech,
            'word_vector': processedtext,
            'lexicon_analysis': lexicon_analysis,
            'polarity_shift_word': polarity_shift_word,
            }

    df = pd.DataFrame(data, columns=['part_of_speech', 'word_vector', 'lexicon_analysis', 'polarity_shift_word'])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df, sentiment, test_size=1 - TRAIN_SIZE, random_state=0)

    print("TRAIN size:", len(X_train))
    print("TEST size:", len(X_test))

    tweet_vectoriser_Bow = CountVectorizer(ngram_range=(1, 1))
    tweet_vectoriser_TFIDF_no_ngram = TfidfVectorizer(ngram_range=(1, 1))
    tweet_vectoriser_TFIDF_with_ngram = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    pos_vectoriser = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

    X_train_BoW = vectorize(X_train, tweet_vectoriser_Bow)
    X_train_TFIDF_no_ngram = vectorize(X_train, tweet_vectoriser_TFIDF_no_ngram)
    X_train_TFIDF_with_ngram = vectorize(X_train, tweet_vectoriser_TFIDF_with_ngram)

    print(f'Bag of Words Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_Bow.get_feature_names()) + 2)
    print(f'TFIDF Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_TFIDF_no_ngram.get_feature_names()) + 2)
    print(f'TFIDF_ngrams Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser_TFIDF_with_ngram.get_feature_names()) + 2)

    # transform tes data set
    X_test_BoW = transform(X_test, tweet_vectoriser_Bow)
    X_test_TFIDF_no_ngram = transform(X_test, tweet_vectoriser_TFIDF_no_ngram)
    X_test_with_ngram = transform(X_test, tweet_vectoriser_TFIDF_with_ngram)

    save_models(X_train_BoW, X_test_BoW, 'BoW')
    save_models(X_train_TFIDF_no_ngram, X_test_TFIDF_no_ngram, 'TFIDF_no_ngram')
    save_models(X_train_TFIDF_with_ngram, X_test_with_ngram, 'TFIDF_with_ngram')

    # now word embedded vectors
    print("Now training word vector model")
    word2vec_model = create_word2vec(df['word_vector'])
    wordvec_arrays = np.zeros((len(processedtext), 200))
    for i in range(len(processedtext)):
        wordvec_arrays[i, :] = create_tweet_vector(processedtext[i], 200, word2vec_model)
    wordvec_df = pd.DataFrame(wordvec_arrays)

    wordvec_df_train = wordvec_df.loc[X_train.index]
    wordvec_df_test = wordvec_df.loc[X_test.index]

    save_models(wordvec_df_train, wordvec_df_test, 'word_vec')

    # save the models for later use
    file = open('vectoriser.pickle', 'wb')
    vectorisers = [tweet_vectoriser_Bow, tweet_vectoriser_TFIDF_no_ngram, tweet_vectoriser_TFIDF_with_ngram]
    pickle.dump(vectorisers, file)
    file.close()

    file = open('y_train.pickle', 'wb')
    pickle.dump(y_train, file)
    file.close()

    file = open('y_test.pickle', 'wb')
    pickle.dump(y_test, file)
    file.close()

    print(f'Dataset processing complete. Time Taken: {round(time.time() - t)} seconds')

    # Plotting the distribution for dataset.
    # make_graphs(processedtext, dataset, X_train, X_test)
    # text analysis
    # get_wordcloud(processedtext)


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
    # doc_vec = text_vectoriser.fit_transform(data['word_vector'])
    # df = pd.DataFrame(doc_vec.toarray().transpose(), index=text_vectoriser.get_feature_names())
    # ex = df.iloc[:,0]
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
    """https://www.kaggle.com/nitin194/twitter-sentiment-analysis-word2vec-doc2vec"""
    tokenized_tweet = preprocessed.apply(lambda x: x.split())  # tokenizing

    model_w2v = gensim.models.Word2Vec(
        tokenized_tweet,
        size=200,  # desired no. of features/independent variables
        window=5,  # context window size
        min_count=2,  # Ignores all words with total frequency lower than 2.
        sg=1,  # 1 for skip-gram model
        hs=0,
        negative=10,  # for negative sampling
        workers=32,  # no.of cores
        seed=34
    )

    model_w2v.train(tokenized_tweet, total_examples=len(preprocessed), epochs=20)
    return model_w2v


def create_tweet_vector(tokens, size, model_w2v):
    """
    Since our data contains tweets and not just words, we’ll have to figure out a way to use the word vectors from
    word2vec model to create vector representation for an entire tweet. There is a simple solution to this problem,
    we can simply take mean of all the word vectors present in the tweet. The length of the resultant vector will be
    the same, i.e. 200. We will repeat the same process for all the tweets in our data and obtain their vectors.
    Now we have 200 word2vec features for our data.

    We will use the below function to create a vector for each tweet by taking the average of the vectors of the
    words present in the tweet.
    :return:
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


def preprocess(textdata):
    """
    CREDIT Nikit Periwal https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :param textdata:
    The Preprocessing steps taken are:

    Lower Casing: Each text is converted to lowercase.
    Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
    Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning.
    (eg: ":)" to "smile")
    Removing Usernames: Replace @Usernames with nothing.
    Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
    Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
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
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    processedText = []
    part_of_speech = []
    lexicon_analysis = []
    polarity_shift_word = []
    longest = 0

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
    seqDigit = r"[0-9]+"
    i = 0
    for tweet in textdata:
        print("tweet #" + str(i))
        score = analyser.polarity_scores(tweet)
        lexicon_analysis.append(score['compound'])
        tweet = tweet.lower()
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "emoji_" + emojis[emoji])  # "EMOJI" + emojis[emoji])
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace @USERNAME to 'USER'. Actually just remove it.
        tweet = re.sub(userPattern, '', tweet)
        # Replace most non alphabets.
        tweet = re.sub(maxAlphaPattern, " ", tweet)
        # todo : remove all words of 3 letters or less?

        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        # separate numbers and letters
        tweet = re.sub(splitDigitChar, r" \1 ", tweet)
        # replace numbers with 'number'
        # tweet = re.sub(seqDigit, "number", tweet)
        polarity = 0
        if 'not' in tweet or 'but' in tweet:
            polarity = 1

        # probably going to stem first, then lemmatize. try both and see results?
        processed, pos = process_sentence(tweet.split())
        processedText.append(processed)
        part_of_speech.append(pos)
        # lexicon_analysis.append(lexicon)
        polarity_shift_word.append(polarity)
        if len(tweet.split()) > longest:
            longest = len(tweet.split())
        i += 1
    return part_of_speech, processedText, lexicon_analysis, polarity_shift_word, longest


def process_sentence(tokens):
    # stopwordlist = set(stopwords.words("english"))
    stopwordlist = ['the', 'of']
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
                    pos = 'r'  # adjverb
                else:
                    pos = 'o'  # other

                if pos in ['n', 'v', 'a', 'r']:
                    word = lemmatizer.lemmatize(word, pos)
                else:
                    word = lemmatizer.lemmatize(word)
                # now stem
                word = stemmer.stem(word)
                processed_sentence.append(word)
                partofspeech.append(pos)

    final_text = ' '.join(processed_sentence)
    final_pos = ' '.join(partofspeech)
    return final_text, final_pos


def get_wordcloud(processedtext):
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


def make_graphs(processedtext, dataset, X_train, X_test):
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
    read()
