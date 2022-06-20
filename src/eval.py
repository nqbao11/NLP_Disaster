import argparse
import subprocess
import os


if __name__ == "__main__":
    #Define argument
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--model", required=True, help="model directory")
    parser.add_argument("--data", required=True, help="evaluation data directory")
    parser.add_argument("-o", "--output",required=True, help="Output directory")

    args = parser.parse_args()
    
    

    #Call for evaluation
    subprocess.run(["python", os.path.join(args.model, "infer.py"), 
                    "--data", args.data, 
                    "--output", args.output])