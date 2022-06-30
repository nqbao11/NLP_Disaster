import main
import os

# import config
from flask import Flask, render_template, request  # import
import cgi

# import PIL

app = Flask(__name__)

UPLOAD_FOLDER = "static"  #
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    full_filename = os.path.join(app.config["UPLOAD_FOLDER"], "t1.jpg")
    return render_template("home.html", img_path=full_filename)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        tweet = request.form
        with open("input_tweet.txt", "w") as f:
            for key, value in tweet.items():
                f.write(value)
        return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
