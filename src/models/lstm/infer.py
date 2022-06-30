import argparse
import os

import torch
from lstm_classifier import LstmClassifier, predict, load_vocab, get_data_test
from config import *


# Later change this init part
# Define hyperparameters for initializing a model instance
# size_of_vocab = len(text_torch.vocab)
size_of_vocab = 3567  # This needs to be changed - vocab size varied based on input - need to get from generate model.py
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
    model = LstmClassifier(
        size_of_vocab,
        embedding_dim,
        num_hidden_nodes,
        num_output_nodes,
        num_layers,
        bidirectional=bidirection,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def infer_file(model, data_test_path, output_file_path):
    text_torch = load_vocab(vocab_path_load)
    tweet_test_df, tweets_for_test = get_data_test(data_test_path)

    # The list for storing predictions
    y_hat = [predict(model, text_torch, sentence) for sentence in tweets_for_test]

    # Create submission file
    submission_df = tweet_test_df[["id"]].copy()
    submission_df["target"] = y_hat
    submission_df.loc[-1] = ["id", "target"]  # Adding a row
    submission_df.index = submission_df.index + 1  # Shifting index
    submission_df.sort_index(inplace=True)
    submission_df.to_csv(output_file_path, header=False, index=False)

def infer_single(model, input_file, output_file):
    with open(input_file,'r') as f:
        input = f.read()
    text_torch = load_vocab(vocab_path_load)
    output = predict(model, text_torch, input)
    with open(output_file, "w") as f:
        f.write(str(output))

if __name__ == "__main__":

    # Parse argument from CMD
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--data", help="evaluation data directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--mode", default="single", help="Inference Mode")
    args = parser.parse_args()

    # Get current directory of file
    current_dir = os.path.dirname(__file__)  # call python folder
    model_name = current_dir.split("\\")[-1]

    model = load_model(os.path.join(current_dir, model_path))

    # Inference
    if args.mode == "single":
        infer_single(model, args.data, args.output )
    else:
        output = os.path.join(args.output, model_name + "_submit.csv")  # need a config file
        infer_file(model, args.data, output)
