from __future__ import print_function

import timeit


class Timer:
    def __init__(self):
        self.times = [timeit.default_timer()]

    def elapsed(self):
        return timeit.default_timer() - self.times[0]

    def split(self):
        self.times.append(timeit.default_timer())
        return self.times[-1] - self.times[0]
