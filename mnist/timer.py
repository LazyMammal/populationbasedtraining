from __future__ import print_function

import timeit


class Timer:
    def __init__(self):
        self.start_time = timeit.default_timer()
        self.split_time = self.start_time

    def elapsed(self):
        return timeit.default_timer() - self.start_time

    def split(self):
        time_now = timeit.default_timer()
        delta_time = time_now - self.split_time
        self.split_time = time_now
        return delta_time
