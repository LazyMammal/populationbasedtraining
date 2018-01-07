def overfit_score(train_accuracy, test_accuracy):
    test_error = 1 - test_accuracy
    train_error = 1 - train_accuracy
    return 1.0 - (train_error / test_error)
