import os
from statistics import mode
from unittest import result
from flask import Flask, render_template_string, request, render_template
import subprocess
ignore = {"template"}

extend = "{% extends 'home.html' %}"
model_option = """
{% block option %}
    {% for model in models %}
        <option value="{{model}}">{{model}}</option>
    {% endfor %}
{% endblock %}
"""
notDisaster = """
{% block output %}
    <h1><span class="badge badge-success">Phewww, not about disaster</span></h1>
{% endblock %}
"""

isDisaster = """
{% block output %}
    <h1><span class="badge badge-danger">Run!</span></h1>
{% endblock %}
"""
app = Flask(__name__)

models = [f for f in os.listdir("./models") if f not in ignore]


@app.route("/", methods = ["GET"])
def home():
    return render_template_string(extend + model_option,models = models)

@app.route("/infer", methods= ["POST"])
def infer():
    model_selected = request.form['model']
    tweet = request.form['tweet']
    with open("tmp_input.txt", "w") as f:
        f.write(tweet)
    subprocess.run([
        "python", "infer.py",
        "--model", model_selected
    ])
    with open("tmp_output.txt", "r") as f:
        result = f.read()
    print(model_selected)
    print(tweet)
    return render_template_string("""
    {% extends 'home.html' %}
    {% block option %}
    {% for model in models %}
        {% if model == model_selected %}
            <option value="{{model}} selected ">{{model}}</option>
        {% endif %}
    {% endfor %}
    {% for model in models %}
        {% if model != model_selected %}
            <option value="{{model}}">{{model}}</option>
        {% endif %}
    {% endfor %}
    {% endblock %}
    {% block text %}{{tweet}}{% endblock %}
    {% block output %}
        {% if result == "0" %}
            <h1><span class="badge badge-success">Phewww, not about disaster</span></h1>
        {% endif %}   
        {% if result != "0" %}
            <h1><span class="badge badge-danger">Run!</span></h1>
        {% endif %}   
    {% endblock %}
    """, models = models, model_selected = model_selected, tweet = tweet, result= result)
if __name__ == "__main__":
    app.run(debug=True)