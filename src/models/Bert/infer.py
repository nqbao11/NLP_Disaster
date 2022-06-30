import argparse
from ast import arg
from operator import mod
import os
from config import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv
from datasets import load_dataset

def load_model(checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


@torch.no_grad()
def infer_file(model, tokenizer, data_file, output, device):
    # model = model.to(device)
    
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

def infer_single(model, tokenizer, input_file, output_file, device):
    with open(input_file, 'r') as f:
        input = f.read()
    encoded = tokenizer(input, return_tensors = 'pt').to(device)
    output = torch.argmax(model(**encoded)[0]).item()
    with open(output_file, 'w') as f:
        f.write(str(output))
if __name__ == "__main__":

    #Parse argument from CMD
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--data", help="evaluation data directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--mode", default="single", help="Inference Mode")
    args = parser.parse_args()

    #Get current directory of file
    current_dir = os.path.dirname(__file__)
    model_name = current_dir.split("\\")[-1]
    #Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(os.path.join(current_dir, "checkpoint"))
    model = model.to(device)
    #Inference
    if args.mode == "single":
        infer_single(model, tokenizer, args.data, args.output, device=device)
    else:
        output = os.path.join(args.output, model_name + "_submit.csv")
        infer_file(model, tokenizer, args.data, output, device=device)
