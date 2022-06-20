from utils import build_freqs, extract_features, get_data_train, gradient_descent, predict_tweet, get_data_test
import numpy as np
import pickle


FILE_NAME_TRAIN = 'train.csv'
FILE_NAME_TEST = 'test.csv'
FILE_NAME_SUBMISSION = 'submission.csv'
MODEL_PATH_SAVE = 'model_logistic_regression'
INPUT_SHAPE = (3, 1)
LEARNING_RATE = 1e-9
NUMBER_OF_ITERATIONS = 500
THRESHOLD = 0.5

class LogisticRegressionClassifier:
    # Constructor
    def __init__(self):
        self.tweets = 'sample tweet'
        self.freqs = {}
        self.theta = np.array([0.1, 0.1, 0.1])
        self.train_x, self.train_y = get_data_train(FILE_NAME_TRAIN)
        return None
  
    # Implement architecture of the model
    def build(self):
        return None

    # Train the defined model with training and validation data 
    def train(self):
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
    def save(self):
        with open(MODEL_PATH_SAVE, 'wb') as f:
            pickle.dump(self, f)
        return None
  

    # Load pretrained model from file
    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self = pickle.load(f)
        return self
    
    
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
        self.tweets = tweets
        y_hat = self.get_label(self.tweets)
        return y_hat


    def test(self):
        # Predict tweets in test.csv from Kaggle

        # Load the data from test.csv into tweet_test_df
        # and extract the 'text' field for testing
        tweet_test_df, tweets_for_test = get_data_test(FILE_NAME_TEST)
            
        # The list for storing predictions
        y_hat = self.get_label(tweets_for_test)

        # Create submission file
        return_df = tweet_test_df[['id']].copy()
        return_df['target'] = y_hat
        return_df.loc[-1] = ['id', 'target']  # Adding a row
        return_df.index = return_df.index + 1  # Shifting index
        return_df.sort_index(inplace=True) 
        return_df.to_csv(FILE_NAME_SUBMISSION, header=False, index=False)

