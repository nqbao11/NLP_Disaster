# Error analysis

print('Label Predicted Tweet\n')
count_pos = 0
count_neg = 0
for x,y in zip(val_x,val_y):
    y_hat = predict_tweet(x, freqs, theta)
    if y == 1:
        count_pos += 1 
    else:
        count_neg += 1

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
        print('\n')
        
print('\nPositive tweet that are classified as negative:', count_pos)
print('\nNegative tweet that are classified as positive:', count_neg)