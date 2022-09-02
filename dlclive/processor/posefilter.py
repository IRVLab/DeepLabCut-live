""""
IRVLab GPLV3

A low pass filter for pose estimators.

"""

import math
import numpy as np
from dlclive.processor import Processor

"""
OneEuroFilter from https://github.com/jaantollander/OneEuroFilt
"""

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class PoseFilter(Processor):
    def __init__(self, minimum_cutoff=0.005, beta=0.7, **kwargs,):
        super().__init__(**kwargs)
        self.min_cutoff = minimum_cutoff
        self.beta = beta
        self.filters = list()

    def process(self, pose, **kwargs):
        time = kwargs['frame_time']

        for k, bp in enumerate(pose):
            try:
                x_filter, y_filter = self.filters[k]
                pose[k][0] = x_filter(time, bp[0])
                pose[k][1] = y_filter(time, bp[1])

            # We hit this if there's no initialized filter.
            except IndexError:
                self.filters.append((OneEuroFilter(time, bp[0], self.min_cutoff, self.beta), OneEuroFilter(time, bp[1], self.min_cutoff, self.beta)))

        return pose
