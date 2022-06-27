from lstm_classifier import LstmClassifier, binary_accuracy, train, evaluate, predict, save_vocab
import torch
import torch.optim as optim
import torch.nn as nn






####### utils.py  #######
import torch 
from torchtext.legacy import data
from nltk.tokenize import TweetTokenizer   # Module for tokenizing strings
import random
from config import *



# Tokenize tweets
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                            reduce_len=True)

# Define a tokenizer
tokenize = lambda s : tokenizer.tokenize(s)


# https://torchtext.readthedocs.io/en/latest/data.html#field
# torchtext.data.Fields holds a Vocab object
TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [(None, None), (None, None), (None, None), ('text', TEXT), ('target', LABEL)]

# Loading custom dataset
training_data = data.TabularDataset(path=FILE_NAME_TRAIN, format='csv', fields=fields, skip_header = True)

# Help reproduce same results
SEED = 2022
# Torch
torch.manual_seed(SEED)

train_data, valid_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))

print('Building vocab...')
TEXT.build_vocab(train_data, min_freq=3, vectors='charngram.100d')
LABEL.build_vocab(train_data)

# Save Vocab after building it
# Will be used later for predicting 
save_vocab(TEXT, VOCAB_PATH)

# Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), 
                                                            batch_size = BATCH_SIZE,
                                                            sort_key = lambda x: len(x.text),
                                                            sort_within_batch=True)





###    end of utils.py  ################################################################3


# Define hyperparameters for training the model
N_EPOCHS = 5
best_valid_loss = float('inf')


# Define hyperparameters for initializing a model instance
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

print('Initializing the model...')
model = LstmClassifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                       bidirectional=bidirection, dropout=dropout)

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
# Alternative for pretrained_embedding: model.embedding.weight.data.copy_(vocab.vectors)


# Define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

# Push to cuda if available
# model = model.to(device)
# criterion = criterion.to(device)

print('Entering training phrase...')
for epoch in range(N_EPOCHS):
     
    # Train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    # Evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    # Save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_lstm.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')





