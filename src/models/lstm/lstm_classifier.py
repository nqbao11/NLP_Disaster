# [Reference] https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/

import torch
import torch.nn as nn
import pandas as pd
import dill  # Similar to pickle

# Use dill instead of pickle because pickle cannot serialize torchtext.legacy.data.Field

from nltk.tokenize import (
    TweetTokenizer,
)  # A combination of hashtag + word is treated as one token instead of two


class LstmClassifier(nn.Module):
    # Define all the layers used in model
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
    ):
        # Constructor
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        # Fully-Connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):  # text = [batch_size, sent_len]
        embedded = self.embedding(text)  # embedded = [batch_size, sent_len, emb_dim]
        # Packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Concat the final forward and backward hidden state
        hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1
        )  # hidden = [batch_size, hid_dim * num_directions]
        dense_outputs = self.fc(hidden)
        # Final activation function
        outputs = self.act(dense_outputs)
        return outputs


# Define metric
def binary_accuracy(preds, y):
    # Round predictions to the closest integer (either 0 or 1)
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    # Set the model in training phase
    model.train()
    for batch in iterator:
        # Resets the gradients after every batch
        optimizer.zero_grad()
        # Retrieve text and no. of words
        text, text_lengths = batch.text
        # Convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()
        # Compute the loss
        loss = criterion(
            predictions, batch.target
        )  # batch.target instead of batch.label?
        # Compute the binary accuracy
        acc = binary_accuracy(predictions, batch.target)
        # Backpropage the loss and compute the gradients
        loss.backward()
        # Update the weights
        optimizer.step()
        # Loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    # Deactivating dropout layers
    model.eval()
    # Deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # Retrieve text and no. of words
            text, text_lengths = batch.text
            # Convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()
            # Compute loss and accuracy
            loss = criterion(predictions, batch.target)
            acc = binary_accuracy(predictions, batch.target)
            # Keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Save vocab object (torchtext.data.Field) to file
def save_vocab(vocab_obj, vocab_path):
    with open(vocab_path, "wb") as f:
        dill.dump(vocab_obj, f)
    return None


# Load vocab object (torchtext.data.Field) from file
def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = dill.load(f)
    return vocab  # need this line


def get_data_test(data_test_path):
    """Load testing data
    Return a dataframe and a list of sentences for testing
    """
    tweet_test_df = pd.read_csv(
        data_test_path,
        header=None,
        usecols=[0, 3],
        names=["id", "text"],
        engine="python",
    )

    # Skip the first row
    tweet_test_df = tweet_test_df.iloc[1:]
    tweets_for_test = tweet_test_df["text"].values.tolist()
    return tweet_test_df, tweets_for_test


# Define tokenizer for predict()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def predict(
    model, text_torch, sentence
):  # text_torch is a torchtext.legacy.data.Field object containing training vocab
    # tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    tokenized = [tok for tok in tokenizer.tokenize(sentence)]  # tokenize the sentence
    indexed = [
        text_torch.vocab.stoi[t] for t in tokenized
    ]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    # tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = model(tensor, length_tensor)  # prediction
    prediction = torch.round(torch.tensor(prediction))
    prediction = prediction.type(torch.int64)
    return prediction.item()
