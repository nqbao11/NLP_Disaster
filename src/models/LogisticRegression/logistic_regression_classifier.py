from utils import build_freqs, extract_features, get_data_train, gradient_descent, predict_tweet, get_data_test
from config import *
import numpy as np
import pickle


class LogisticRegressionClassifier:
    # Constructor
    def __init__(self, file_name_train):
        self.tweets = 'sample tweet'
        self.freqs = {}
        self.theta = np.array([0.1, 0.1, 0.1])

        return None
  
    # Implement architecture of the model
    def build(self):
        return None

    # Train the defined model with training and validation data 
    def train(self, file_name_train):
        train_x, train_y = get_data_train(file_name_train)
        # Create the frequency dictionary
        self.freqs = build_freqs(self.train_x, self.train_y)

        # TRAIN THE MODEL
        # Collect the features 'x' and stack them into a matrix 'X'
        X = np.zeros((len(self.train_x), 3))
        for i in range(len(self.train_x)):
            X[i, :] = extract_features(self.train_x[i], self.freqs)

        # Training labels corresponding to X
        Y = self.train_y

        # Apply gradient descent
        cost, theta = gradient_descent(X, Y, np.zeros(INPUT_SHAPE), LEARNING_RATE, NUMBER_OF_ITERATIONS)
        self.theta = theta
        return None
  
    # Save the trained model to a file
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        return None
  

    # Load pretrained model from file
    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self = pickle.load(f)
    
    
    # Visualize the model
    def summary(self):
        return None
  

    def get_label(self, tweets):
        if type(tweets) is list:  # tweets is a list of strings
            y_hat = []
            for tweet in tweets:
                # Get the probability prediction for the tweet
                y_probability = predict_tweet(tweet, self.freqs, self.theta)
                # Convert probability to label based on a threshold
                if y_probability > THRESHOLD:
                    y_hat.append(1)
                else:
                    y_hat.append(0)
            return y_hat

        elif type(tweets) is str:  # tweets is a string
            # Initialize prediction to None
            y_hat = None
            # Get the label prediction for the tweet
            y_probability = predict_tweet(tweets, self.freqs, self.theta)
            # Compare prediction probability with a threshold to get the label
            if y_probability > THRESHOLD:
                y_hat = 1
            else:
                y_hat = 0

            return y_hat


    # Apply the model with new input data
    # for input is a list of tweets
    def predict(self, tweets):
        y_hat = self.get_label(tweets)
        return y_hat


    def test(self, test_file, output_file):
        # Predict tweets in test.csv from Kaggle

        # Load the data from test.csv into tweet_test_df
        # and extract the 'text' field for testing
        tweet_test_df, tweets_for_test = get_data_test(test_file)
            
        # The list for storing predictions
        y_hat = self.get_label(tweets_for_test)

        # Create submission file
        submission_df = tweet_test_df[['id']].copy()
        submission_df['target'] = y_hat
        submission_df.loc[-1] = ['id', 'target']  # Adding a row
        submission_df.index = submission_df.index + 1  # Shifting index
        submission_df.sort_index(inplace=True) 
        submission_df.to_csv(output_file, header=False, index=False)

