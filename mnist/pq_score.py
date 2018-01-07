def pq_accuracy(train_avg_accuracy, train_best_accuracy, test_avg_accuracy, test_best_accuracy):
    return pq_error(1.0 - train_avg_accuracy,
                    1.0 - train_best_accuracy,
                    1.0 - test_avg_accuracy,
                    1.0 - test_best_accuracy)


def pq_error(train_avg_error, train_best_error, test_avg_error, test_best_error):
    gl = gl_error(test_avg_error, test_best_error)
    p = p_error(train_avg_error, train_best_error)
    return gl / p


def gl_accuracy(test_avg_accuracy, test_best_accuracy):
    return gl_error(1.0 - test_avg_accuracy,
                    1.0 - test_best_accuracy)


def gl_error(test_avg_error, test_best_error):
    return test_avg_error / test_best_error


def p_accuracy(train_avg_accuracy, train_best_accuracy):
    return p_error(1.0 - train_avg_accuracy,
                   1.0 - train_best_accuracy)


def p_error(train_avg_error, train_best_error):
    return train_avg_error / train_best_error
