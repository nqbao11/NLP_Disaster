import argparse
import os

import torch
from lstm_classifier import LstmClassifier, predict, load_vocab, get_data_test
from config import *
import dill


# Later change this init part
# Define hyperparameters for initializing a model instance
# size_of_vocab = len(TEXT.vocab)
size_of_vocab = 3567  # This need to be changed - vocab size varied based on input - need to get from generate model.py
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2


def load_model(model_path):
    """
    Load model from a binary file
    """
    model = model = LstmClassifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                       bidirectional=bidirection, dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def infer(model, data_test, output_file):    ######## THE LAST PART to complete  
    # extract strings from data_test, use predict() to get the result and write to output_file
    TEXT = load_vocab(VOCAB_PATH_LOAD)
    tweet_test_df, tweets_for_test = get_data_test(data_test)  # data_test == FILE_NAME_TEST; just want to keep the template unchanged

    # The list for storing predictions
    y_hat = [predict(model, TEXT, sentence) for sentence in tweets_for_test]

    # Create submission file
    submission_df = tweet_test_df[['id']].copy()
    submission_df['target'] = y_hat
    submission_df.loc[-1] = ['id', 'target']  # Adding a row
    submission_df.index = submission_df.index + 1  # Shifting index
    submission_df.sort_index(inplace=True) 
    submission_df.to_csv(output_file, header=False, index=False)





if __name__ == "__main__":

    #Parse argument from CMD
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--data", help="evaluation data directory")
    parser.add_argument("-o", "--output", help="Output directory")

    args = parser.parse_args()

    #Get current directory of file
    current_dir = os.path.dirname(__file__)  # call python folder
    model_name = current_dir.split("\\")[-1]

    model = load_model(os.path.join(current_dir, MODEL_PATH))

    #Inference
    output = os.path.join(args.output, model_name + "_submit.csv")  # need a config file
    infer(model, args.data, output)
