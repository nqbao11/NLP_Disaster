import argparse
import os
# from config import *
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# import csv
# from datasets import load_dataset

from logistic_regression_classifier import LogisticRegressionClassifier
from config import *

def load_model(model_path):
    model = LogisticRegressionClassifier(os.path.join(current_dir, FILE_NAME_TRAIN))
    model.load(model_path)
    return model

def infer_file(model, data_test, output_file):
    model.test(data_test, output_file)

def infer_single(model, input_file, output_file):
    with open(input_file,'r') as f:
        input = f.read()
    output = model.predict(input)
    with open(output_file, "w") as f:
        f.write(str(output))

if __name__ == "__main__":

    #Parse argument from CMD
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--data", help="evaluation data directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--mode", default="single", help="Inference Mode")
    args = parser.parse_args()

    #Get current directory of file
    current_dir = os.path.dirname(__file__)  # call python folder
    model_name = current_dir.split("\\")[-1]

    classifier = load_model(os.path.join(current_dir, MODEL_PATH))

    #Inference
    if args.mode == "single":
        infer_single(classifier, args.data, args.output)
    else:
        output = os.path.join(args.output, model_name + "_submit.csv")  # need a config file
        infer_file(classifier, args.data, output)
