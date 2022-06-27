# A tiny NLP project 

## About
 - A project about predicting whether a Tweet are about real disasters or not.
 - Dataset: The dataset of this project is the dataset used for <a href="https://www.kaggle.com/competitions/nlp-getting-started/overview">this Kaggle competition</a>.
 - Define of Done: Get 1.00 Accuarcy score on Kaggle private test (Current Top1 have also 1.00 accuracy score)
## Project Structure

## Conventions
|Name|Docs|
|-|-|
|Code convention|https://peps.python.org/pep-0008/|
|Commit convention |Does this matter?|

## Result
> Note: These experiments are evaluated over a blind test dataset of 3262 tweets, submitted on Kaggle.

|Model|Experiment|Result in private test|Documents|
|-|-|-|-|
|Logistic Regression|Preprocess tweet|0.73|.|
|LSTM|Moses Tokenizer|0.73|.|
|LSTM|Tweet Tokenizer|0.79|.|
|Bert base|Not Preprocess tweet| 0.83|.|

## Commands
`python src\eval.py --model src\models\logistic_regression --data data\test.csv --output results`
`python src\eval.py --model src\models\lstm --data data\test.csv --output results`

