import argparse
import os
from config import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv
from datasets import load_dataset

def load_model(checkpoint):
    """
    Load model from a binary file
    """
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


@torch.no_grad()
def infer(model, tokenizer, data_file, output, device):
    model = model.to(device)
    
    fout = open(output, "w")
    writer = csv.writer(fout)
    writer.writerow(["id", "target"])
    
    dataset = load_dataset("csv", data_files=data_file)
    for data in dataset["train"]:
        id = data["id"]
        encoded = tokenizer(data['text'], return_tensors = 'pt').to(device)
        predict = torch.argmax(model(**encoded)[0]).item()
        writer.writerow([id, 1])
    
    fout.close()

if __name__ == "__main__":

    #Parse argument from CMD
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--data", help="evaluation data directory")
    parser.add_argument("-o", "--output", help="Output directory")

    args = parser.parse_args()

    #Get current directory of file
    current_dir = os.path.dirname(__file__)
    model_name = current_dir.split("\\")[-1]
    #Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(os.path.join(current_dir, checkpoint))

    #Inference
    output = os.path.join(args.output, model_name + "_submit.csv")
    infer(model, tokenizer, args.data, output, device=device)
