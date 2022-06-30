"""
FineTuning Bert from pretrained
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_metric, load_dataset, Dataset
import numpy as np
import regex as re
import pandas as pd
def load_model(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return tokenizer, model

def pre_process(tweet):
    tweet = re.sub(r'\$\w*', '', tweet)

    tweet = re.sub(r'^RT[\s]+', '', tweet)

    tweet = re.sub(r'http[s]*:\/\/[\w,\d,.,\/,]*', '', tweet)

    tweet = re.sub(r'#[\w,\d]*', '', tweet)

    tweet = re.sub(r'@[\w,\d]*', '', tweet)

    return tweet 

def load_data(path):
    train  = pd.read_csv(path).drop(['keyword', 'location'],1).set_index('id')
    return train

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    model, tokenizer = load_model("bert-base-uncased")

    train = load_data(...)
    train['text'] = train['text'].map(lambda sentence: pre_process(sentence))
    dataset = Dataset.from_pandas(train)

    dataset = dataset.rename_columns({
        "target": "label",
        "text": "sentence"
    })
    dataset = dataset.train_test_split(test_size=0.1)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    metric = load_metric('glue', 'sst2')

    metric_name = 'accuracy'
    args = TrainingArguments(
        "bert-disaster",
        per_device_train_batch_size=32,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        metric_for_best_model =metric_name,
        learning_rate=1e-5
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()