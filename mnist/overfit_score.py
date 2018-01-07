def overfit_accuracy(train_accuracy, test_accuracy):
    test_error = 1 - test_accuracy
    train_error = 1 - train_accuracy
    return 1.0 - (train_error / test_error)


def overfit_blended(train_accuracy, test_accuracy):
    """
    strike a balance between 'overfit' and test accuracy
    """
    return (1.0 - overfit_accuracy(train_accuracy, test_accuracy)) * test_accuracy
