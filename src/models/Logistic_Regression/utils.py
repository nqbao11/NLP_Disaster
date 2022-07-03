# [Reference] https://www.kaggle.com/code/notrealli/coursera-nlp-1-1-1-preprocessing

import nltk  # Python library for NLP
from nltk.corpus import twitter_samples  # Sample Twitter dataset from NLTK
import matplotlib.pyplot as plt  # Library for visualization
import random  # Pseudo-random number generator
import numpy as np
import pandas as pd
import re  # Library for regular expression operations
import string  # For string operations

from nltk.corpus import stopwords  # Module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # Module for stemming
from nltk.tokenize import TweetTokenizer  # Module for tokenizing strings
import pickle  # Module for save and load the model


def get_data_train(FILE_NAME_TRAIN):
    # Training data provided by Kaggle has 5 columns, only 2 in 5 columns will
    # be used to train the Logistic Regression classifier.
    column_names = ["text", "target"]

    tweet_df = pd.read_csv(
        FILE_NAME_TRAIN, header=None, usecols=[3, 4], names=column_names
    )
    tweet_df = tweet_df.iloc[1:]  # skip the header

    # Select sets of positive and negative tweets
    positive_tweets = tweet_df.loc[tweet_df["target"] == "1"]["text"].values.tolist()
    negative_tweets = tweet_df.loc[tweet_df["target"] == "0"]["text"].values.tolist()

    train_x = positive_tweets + negative_tweets
    train_y = np.append(
        np.ones((len(positive_tweets), 1)), np.zeros((len(negative_tweets), 1)), axis=0
    )

    return train_x, train_y


def get_data_test(FILE_NAME_TEST):
    tweet_test_df = pd.read_csv(
        FILE_NAME_TEST, header=None, usecols=[0, 3], names=["id", "text"]
    )

    # Skip the first row
    tweet_test_df = tweet_test_df.iloc[1:]
    tweets_for_test = tweet_test_df["text"].values.tolist()
    return tweet_test_df, tweets_for_test


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    # Remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # Remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # Remove hyperlinks
    tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet)
    # Remove hashtags, only remove the hash # sign from the word
    tweet = re.sub(r"#", "", tweet)
    # Tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (
            word not in stopwords_english
            and word not in string.punctuation  # Remove stopwords
        ):  # Remove punctuation
            stem_word = stemmer.stem(word)  # Stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP - No Operation - if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


def sigmoid(z):
    """
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    """
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    return h


def predict_tweet(tweet, freqs, theta):
    """
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: a vector of weights with shape (3,1)
    Output: 
        y_pred: the probability of a tweet being positive or negative
    """

    # Extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # Make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


def gradient_descent(x, y, theta, alpha, num_iters):
    """
    Input:
        x: matrix of features which is (m, n+1) n+1 instead of n because of the bias term
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        cost: the final cost
        theta: your final weight vector
    Guidance: print the cost to make sure that it is going down.
    """

    # Get the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):

        # Get the dot product of x and theta
        z = np.dot(x, theta)

        # Get the sigmoid of z
        h = sigmoid(z)

        # Calculate the cost function
        # Note: np.array.transpose() has the same effect as np.array.T
        # np.array.T just makes code a little more readable
        cost = -1.0 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

        # update the weights theta
        theta = theta - (alpha / m) * np.dot(x.T, (h - y))

    cost = float(cost)
    return cost, theta


def extract_features(tweet, freqs):
    """
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    """
    # Pre-process a tweet string: tokenize, stem, and remove stopwords
    word_list = process_tweet(tweet)

    # 3 features of a tweet in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # Set the bias term to 1
    x[0, 0] = 1

    # Loop through each word in the list of words
    for word in word_list:

        # Increment the word count for the positive label 1
        # If the word is not in the in the freqs, increment by 0
        x[0, 1] += freqs.get((word, 1.0), 0)

        # Increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert x.shape == (1, 3)
    return x

