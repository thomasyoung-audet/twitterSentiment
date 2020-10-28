# utilities
import re
import os
import pickle
import numpy as np
import pandas as pd
import time

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


def read():
    """
    Much of this is inspired from Nikit Periwal at
    https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :return:
    """
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    TRAIN_SIZE = 0.8

    dataset_filename = os.listdir("../project/input")[0]
    dataset_path = os.path.join("../project/", "input", dataset_filename)
    print("Open file:", dataset_path)
    dataset = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    # Storing data in lists.
    text, sentiment = list(dataset['text']), list(dataset['sentiment'])
    # Removing the unnecessary columns.
    dataset = dataset[['sentiment', 'text']]
    # Replacing the values to ease understanding.
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
    # Plotting the distribution for dataset.
    ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                                   legend=False)
    ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
    # Storing data in lists.
    text, sentiment = list(dataset['text']), list(dataset['sentiment'])
    fig = ax.get_figure()
    fig.savefig('test.png')

    t = time.time()
    processedtext = preprocess(text)
    print(f'Text Preprocessing complete.')
    print(f'Time Taken: {round(time.time() - t)} seconds')

    # text analysis
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

def preprocess(textdata):
    """
    CREDIT Nikit Periwal https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners#Analysing-the-data
    :param textdata:
    The Preprocessing steps taken are:

    Lower Casing: Each text is converted to lowercase.
    Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
    Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")
    Replacing Usernames: Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")
    Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
    Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
    Removing Short Words: Words with length less than 2 are removed.
    Removing Stopwords: Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
    Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)

    """

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                    'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                    'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                    's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                    'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                    "youve", 'your', 'yours', 'yourself', 'yourselves']

    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()

        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, ' USER', tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            # if word not in stopwordlist:
            if len(word) > 1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')

        processedText.append(tweetwords)

    return processedText


if __name__ == '__main__':
    read()
