# Use the trained model to predict sentiment of a tweet
def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: a vector of weights with shape (3,1)
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # Extract the features of the tweet and store it into x
    x = extract_features(tweet,freqs)
    
    # Make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))    
    return y_pred


def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    
    # The list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # Get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # Append 1 to the list
            y_hat.append(1)
        else:
            # Append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy




# Create the frequency dictionary
freqs = build_freqs(train_x, train_y)


# Train the model
# Collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# Training labels corresponding to X
Y = train_y

# Apply gradient descent
cost, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 600)
print(f"The cost after training is {cost:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


# Predict tweets in test.csv from Kaggle
file_name = 'test.csv'
tweet_test_df = pd.read_csv(file_name, header=None, usecols=[0,3], names=['id', 'text'])


# Skip the first row
tweet_test_df = tweet_test_df.iloc[1:]
tweets_4_test = tweet_test_df['text'].values.tolist()

# Collect the features 'x' and stack them into a matrix 'X'
X_predict = np.zeros((len(tweets_4_test), 3))
for i in range(len(tweets_4_test)):
    X_predict[i, :]= extract_features(tweets_4_test[i], freqs)
    

# The list for storing predictions
y_hat = []

for tweet in tweets_4_test:
    # Get the label prediction for the tweet
    y_pred = predict_tweet(tweet, freqs, theta)

    if y_pred > 0.5:
        # Append 1 to the list
        y_hat.append(1)
    else:
        # Append 0 to the list
        y_hat.append(0)


return_df = tweet_test_df[['id']].copy()
return_df['target'] = y_hat
return_df.loc[-1] = ['id', 'target']  # Adding a row
return_df.index = return_df.index + 1  # Shifting index
return_df.sort_index(inplace=True) 
return_df.to_csv("submission_test_summary_code_2.csv",header=False, index=False)
