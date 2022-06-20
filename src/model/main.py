from logistic_regression_classifier import LogisticRegressionClassifier


def main():
    lr_classifier = LogisticRegressionClassifier()
    # lr_classifier.train()
    # lr_classifier.save()
    # lr_classifier.test()
    lr_classifier = lr_classifier.load('model_logistic_regression')
    trial = []
    trial_1 = lr_classifier.predict("on the outside you're ablaze and alive")
    trial_2 = lr_classifier.predict("hi")
    trial.append(trial_1)
    trial.append(trial_2)
    print('Predicted class(es):', trial)
    print('Parameters of model:', lr_classifier.theta)


if __name__ == "__main__":
    main()
