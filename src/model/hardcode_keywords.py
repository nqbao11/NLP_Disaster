import pandas as pd

FILE_NAME_TEST = 'test.csv'
FILE_NAME_SUBMISSION = 'submission_baseline_for_hardcode.csv'
FILE_NAME_SUBMISSION_HARDCODE = 'submission_hardcode.csv'
POSITIVE_KEYWORDS = ['wreckage', 'debris', 'derailment']
NEGATIVE_KEYWORDS = ['aftershock']

# Extract IDs of tweets whose keywords are either exclusively postive or exclusively negative
def get_ids_and_keywords():
    # Testing data provided by Kaggle has 5 columns, only 2 in 5 columns will
    # be used to extract id and keywords.
    column_names = ['id', 'keyword']

    tweet_df = pd.read_csv(FILE_NAME_TEST, header=None, usecols=[0,1], 
                        names=column_names)
    tweet_df = tweet_df.iloc[1:]  # Skip the header

    positive_tweet_ids = tweet_df.loc[tweet_df['keyword'].isin(POSITIVE_KEYWORDS)]['id'].values.tolist()
    negative_tweet_ids = tweet_df.loc[tweet_df['keyword'].isin(NEGATIVE_KEYWORDS)]['id'].values.tolist()

    return positive_tweet_ids, negative_tweet_ids


def modify_submission(positive_tweet_ids, negative_tweet_ids):
    column_names = ['id', 'target']
    tweet_df = pd.read_csv(FILE_NAME_SUBMISSION, header=None, usecols=[0,1], 
                        names=column_names)
    tweet_df = tweet_df.iloc[1:]  # skip the header

    submission_df = tweet_df.copy()

    check_pos = submission_df.loc[submission_df['id'].isin(positive_tweet_ids)]['id'].values.tolist()
    check_neg = submission_df.loc[submission_df['id'].isin(negative_tweet_ids)]['id'].values.tolist()

    for i in check_pos:
        submission_df.loc[submission_df['id'] == i, 'target'] = 1

    for i in check_neg:
        submission_df.loc[submission_df['id'] == i, 'target'] = 0

    submission_df.loc[-1] = ['id', 'target']  # Adding a row
    submission_df.index = submission_df.index + 1  # Shifting index
    submission_df.sort_index(inplace=True) 
    submission_df.to_csv(FILE_NAME_SUBMISSION_HARDCODE, header=False, index=False)


def main():
    positive_ids, negative_ids = get_ids_and_keywords()
    modify_submission(positive_ids, negative_ids)


if __name__ == "__main__":
    main()
