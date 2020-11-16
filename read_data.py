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
from nltk.corpus import sentiwordnet as swn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion


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
    dataset = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, nrows=1000000)

    # Removing the unnecessary columns.
    dataset = dataset[['sentiment', 'text']]
    # Replacing the values to ease understanding.
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
    # Plotting the distribution for dataset.
    # make_graphs(dataset)
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
    # text analysis
    # get_wordcloud(processedtext)
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

    tweet_vectoriser = CountVectorizer(ngram_range=(1, 2))
    pos_vectoriser = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

    X_train = vectorize(X_train, pos_vectoriser, tweet_vectoriser)
    # X_columns = tweet_vectoriser.get_feature_names() + pos_vectoriser.get_feature_names() + df[
    #     ['lexicon_analysis', 'polarity_shift_word']].columns.tolist()
    # print(X_columns)
    print(f'Vectoriser fitted.')
    print('No. of feature_words: ', len(tweet_vectoriser.get_feature_names()) + 6)

    # transform data set into something that we can train and test against
    X_test = transform(X_test, pos_vectoriser, tweet_vectoriser)

    print(f'Data Transformed.')

    # save the models for later use
    file = open('vectoriser.pickle', 'wb')
    pickle.dump((pos_vectoriser, tweet_vectoriser), file)
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


def vectorize(data, pos_vectoriser, text_vectoriser):
    vectorized_data = sp.sparse.hstack((pos_vectoriser.fit_transform(data['part_of_speech']),
                                        text_vectoriser.fit_transform(data['word_vector']),
                                        data[['lexicon_analysis', 'polarity_shift_word']].values),
                                       format='csr')
    return vectorized_data


def transform(data, pos_vectoriser, text_vectoriser):
    vectorized_data = sp.sparse.hstack((pos_vectoriser.transform(data['part_of_speech']),
                                        text_vectoriser.transform(data['word_vector']),
                                        data[['lexicon_analysis', 'polarity_shift_word']].values),
                                       format='csr')
    return vectorized_data


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

    # stopwordlist = set(stopwords.words("english"))

    processedText = []
    part_of_speech = []
    lexicon_analysis = []
    polarity_shift_word = []
    longest = 0

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    # Create sentiment instensity analyser
    analyser = SentimentIntensityAnalyzer()
    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    maxAlphaPattern = "[^a-zA-Z0-9!?.]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    i = 0
    for tweet in textdata:
        print("tweet #" + str(i))

        score = analyser.polarity_scores(tweet)
        lexicon_analysis.append(score['compound'])

        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "emoji_" + emojis[emoji])  # "EMOJI" + emojis[emoji])
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace @USERNAME to 'USER'. Actually just remove it.
        tweet = re.sub(userPattern, '', tweet)
        # Replace most non alphabets.
        tweet = re.sub(maxAlphaPattern, " ", tweet)
        lexicon = []

        # for word in tweet.split():
        #     if len(word) > 1:
        #         scores = analyser.polarity_scores(word)
        #         lexicon.append(scores['compound'])

        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        tweet = tweet.lower()

        polarity = 0
        if 'not' in tweet or 'but' in tweet:
            polarity = 1

        lemmatized, pos = lemmatize_sentence(tweet.split(), wordLemm)
        processedText.append(lemmatized)
        part_of_speech.append(pos)
        # lexicon_analysis.append(lexicon)
        polarity_shift_word.append(polarity)
        if len(tweet.split()) > longest:
            longest = len(tweet.split())
        i += 1
    return part_of_speech, processedText, lexicon_analysis, polarity_shift_word, longest


def lemmatize_sentence(tokens, lemmatizer):
    lemmatized_sentence = []
    partofspeech = []
    for word, tag in pos_tag(tokens):
        if len(word) > 1:
            if tag.startswith('NN'):
                pos = 'n'  # noun
            elif tag.startswith('VB'):
                pos = 'v'  # verb
            elif tag.startswith('JJ'):
                pos = 'a'  # adjective
            else:
                pos = 'o'  # other
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
            partofspeech.append(pos)
    final_text = ' '.join(lemmatized_sentence)
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
