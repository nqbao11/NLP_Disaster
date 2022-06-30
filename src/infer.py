import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evalution")
    parser.add_argument("--model", required=True, help="model directory")

    args = parser.parse_args()
    subprocess.run([
        "python", os.path.join("models",args.model,"infer.py"),
        "--data", "tmp_input.txt",
        "--output", "tmp_output.txt",
        "--mode", "single"
    ])