def overfit_score(test_accuracy, train_accuracy):
    test_error = 1 - test_accuracy
    train_error = 1 - train_accuracy
    return (test_error - train_error) / test_error
