from sklearn.svm import SVC
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

def training_simple(train_features,
                        train_labels,
                        test_features,
                        test_labels,
                        C=200,
                        tol=1e-3,
                        verbose=True):
    clf = SVC(C=C,
          kernel='rbf',
          tol=tol,
          verbose=verbose,
          max_iter=-1)
    clf.fit(train_features, train_labels)
    training_predictions = clf.predict(train_features)
    test_predictions = clf.predict(test_features)
    correct_train_samples = 0
    for i in range(train_labels.shape[0]):
        if training_predictions[i] == train_labels[i]:
            correct_train_samples +=1
    correct_test_samples = 0
    for i in range(test_labels.shape[0]):
        if test_predictions[i] == test_labels[i]:
            correct_test_samples +=1
    if verbose:
        print(f"Training F1-score: {f1_score(train_labels, training_predictions)}")
        print(f"Training accuracy: {correct_train_samples/train_labels.shape[0]}")
        print(f"Testing F1-score: {f1_score(test_labels, test_predictions)}")
        print(f"Testing accuracy: {correct_test_samples/test_labels.shape[0]}")
    return (clf,
            correct_train_samples/train_labels.shape[0],
            f1_score(train_labels, training_predictions),
            correct_test_samples/test_labels.shape[0],
            f1_score(test_labels, test_predictions))

def cross_validation_training(train_features,
                                train_labels,
                                test_features,
                                test_labels,
                                tol=1e-10,
                                Cs=[0.1, 1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 500, 1000, 2000, 5000, 10000]):
    """
    Test different values of C and returns the best value
    according to the F1-score as well as associated test
    F1-score
    """
    training_accuracies = []
    training_f1_scores = []
    test_accuracies = []
    test_f1_scores = []
    for C_to_test in Cs:
        _, train_acc, train_f1, test_acc, test_f1 = training_simple(train_features,
                                                                    train_labels,
                                                                    test_features,
                                                                    test_labels,
                                                                    C_to_test,
                                                                    tol=tol,
                                                                    verbose=False)
        training_accuracies.append(train_acc)
        training_f1_scores.append(train_f1)
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(Cs, training_accuracies, label="Training accuracy")
    ax1.plot(Cs, test_accuracies, label="Testing accuracy")
    ax1.vlines(Cs[np.argmax(test_f1_scores)], np.amin(test_f1_scores), np.amax(training_f1_scores), linestyles="dashed")
    ax1.set_title("Accuracies over $C$ parameter")
    ax1.legend()
    ax2.plot(Cs, training_f1_scores, label="Training F1-scores")
    ax2.plot(Cs, test_f1_scores, label="Testing F1-scores")
    ax2.vlines(Cs[np.argmax(test_f1_scores)], np.amin(test_f1_scores), np.amax(training_f1_scores), linestyles="dashed")
    ax2.set_title("F1-scores over $C$ parameter")
    ax2.legend()
    plt.show()
    return Cs[np.argmax(test_f1_scores)], np.amax(test_f1_scores)

