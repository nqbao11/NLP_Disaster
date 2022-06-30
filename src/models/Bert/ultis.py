def drop_mislabeled_tweet(data_frame):
    df_mislabeled = data_frame.groupby(['text']).nunique().sort_values(by='target', ascending=False)
    df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
    mislabeled_tweet = set(df_mislabeled.index.to_list())

    for tweet in mislabeled_tweet:
        data_frame.drop(data_frame[data_frame['text'] == tweet].index, inplace=True)
    
    return data_frame
    