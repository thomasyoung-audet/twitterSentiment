# utilities
import re
import os
import pandas as pd
import time
import pickle
# plotting
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def read():
    """
    Much of this is inspired from Nikit Periwal at
    https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :return:
    """
    t = time.time()
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    TRAIN_SIZE = 0.8

    dataset_filename = os.listdir("../project/input")[0]
    dataset_path = os.path.join("../project/", "input", dataset_filename)
    print("Open file:", dataset_path)
    dataset = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    # Removing the unnecessary columns.
    dataset = dataset[['sentiment', 'text']]
    # Replacing the values to ease understanding.
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
    # Plotting the distribution for dataset.
    # make_graphs(dataset)
    # Storing data in lists.
    text, sentiment = list(dataset['text']), list(dataset['sentiment'])

    processedtext, polarity_scores = preprocess(text)
    # save the models for later use
    file = open('processedtext.pickle', 'wb')
    file2 = open('polarity_scores.pickle', 'wb')
    pickle.dump(processedtext, file)
    file.close()
    pickle.dump(polarity_scores, file2)
    file2.close()
    # Load
    file = open('processedtext.pickle', 'rb')
    processedtext = pickle.load(file)
    file.close()
    file = open('polarity_scores.pickle', 'rb')
    polarity_scores = pickle.load(file)
    file.close()
    print(f'Text Preprocessing complete.')
    # text analysis
    # get_wordcloud(processedtext)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size=1 - TRAIN_SIZE,
                                                        random_state=0)
    print("TRAIN size:", len(X_train))
    print("TEST size:", len(X_test))

    # create TF-IDF table # shouldn't this be 248446, the number of unique words?
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)
    print(f'Vectoriser fitted.')
    print('No. of feature_words: ', len(vectoriser.get_feature_names()))

    # transform data set into something that we can train and test against
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    print(f'Data Transformed.')

    # save the models for later use
    file = open('vectoriser.pickle', 'wb')
    pickle.dump(vectoriser, file)
    file.close()

    # save the models for later use
    file = open('X_train.pickle', 'wb')
    pickle.dump(X_train, file)
    file.close()

    # save the models for later use
    file = open('X_test.pickle', 'wb')
    pickle.dump(X_test, file)
    file.close()

    # save the models for later use
    file = open('y_train.pickle', 'wb')
    pickle.dump(y_train, file)
    file.close()

    # save the models for later use
    file = open('y_test.pickle', 'wb')
    pickle.dump(y_test, file)
    file.close()

    print(f'Dataset processing complete. Time Taken: {round(time.time() - t)} seconds')


def preprocess(textdata):
    """
    CREDIT Nikit Periwal https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :param textdata:
    The Preprocessing steps taken are:

    Lower Casing: Each text is converted to lowercase.
    Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
    Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning.
    (eg: ":)" to "smile")
    Replacing Usernames: Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")
    Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
    Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
    Removing Short Words: Words with length less than 2 are removed.
    Removing Stopwords: Stopwords are the English words which does not add much meaning to a sentence. They can safely
    be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
    Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)

    """

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = set(stopwords.words("english"))

    processedText = []
    polarity_score = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    # Create sentiment instensity analyser
    analyser = SentimentIntensityAnalyzer()
    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    i = 1
    for tweet in textdata:
        print("tweet #" + str(i))
        polarity_score.append(analyser.polarity_scores(tweet))
        tweet = tweet.lower()

        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, emojis[emoji])  # "EMOJI" + emojis[emoji])
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, ' USER', tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        words = lemmatize_sentence(tweet.split(), wordLemm, stopwordlist)
        processedText.append(' '.join(words))
        i += 1
    return processedText, polarity_score


def lemmatize_sentence(tokens, lemmatizer, stopwordlist):
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if word not in stopwordlist:
            if len(word) > 1:
                if tag.startswith('NN'):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'
                lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


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


def make_graphs(dataset):
    ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                                   legend=False)
    ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
    # Storing data in lists.
    fig = ax.get_figure()
    fig.savefig('bar_graph.png')
    all_words = []
    for line in list(dataset['text']):
        words = line.split()
        for word in words:
            all_words.append(word.lower())

    plt.figure(figsize=(12, 5))
    plt.title('Top 25 most common words')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(all_words)
    fd.plot(25, cumulative=False)
    plt.savefig('common_words.png')


if __name__ == '__main__':
    read()
