""""
IRVLab GPLV3

A set of filters for pose estimators.

"""

import math
import numpy as np
from dlclive.processor import Processor

"""
OneEuroFilter from https://github.com/jaantollander/OneEuroFilter
"""

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0, pcutoff=0.7):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.pcutoff = pcutoff
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x, conf):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        if conf > self.pcutoff:

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

        else:
            
            return self.x_prev

class MovingAverageFilter:
    def __init__(self, time, pose, pcutoff=0.5, maximum_distance=50, memory_length=5):
        self.pcutoff = pcutoff
        self.maximum_distance = maximum_distance
        self.memory = np.zeros((memory_length, 4))

        # Set initial values of memory, if the estimate is good enough.
        if pose[2] > self.pcutoff:
            self.memory[-1] = [time, pose[0], pose[1], pose[2]]

    def __call__(self, time, pose):
        x,y,conf = pose

        # If we believe the estimate
        if self.pose_acceptable(pose):
            # First, update the memory. We're storing, but not using time. In the future we could use time distance as another 
            self.memory[:-1] = self.memory[1:]
            self.memory[-1] = [time, x, y, conf]

        # If we don't believe in the estimate
        else:
            # Remove the first element, as long as the list isn't empty. 
            # We do this to keep the estimate recent.
            self.memory[:-1] = self.memory[1:]
            self.memory[-1] = [0, 0 ,0, 0]
        
        # If we have more than nothing in the memory, compute the mean.
        if np.any(self.memory):
            # print("Memory at this point: {}".format(self.memory))
            # print("Memory used for mean: {}".format(self.memory[~np.all(self.memory == 0, axis=1), 1:]))

            # Then, compute the average of the memory (only the non-zero rows)
            relevent_memory = self.memory[~np.all(self.memory == 0, axis=1), 1:]
            # We calculate weights by inverse time distance, scaled by confidence.
            weights = abs((relevent_memory[:,0] - time)/max(relevent_memory[:,0]) - 1) * relevent_memory[:,3]
            x_mean, y_mean, conf_mean = np.mean(relevent_memory, axis=0, weights= weights)

            # print("Calculated mean: {}".format([x_mean, y_mean, conf_mean]))
            # Return the average of the memory.
            return [x_mean, y_mean, conf_mean]

        # If we don't have enough in the memory, just return it as is.
        else:
            return [x, y, conf]

    def pose_acceptable(self, pose):
        x,y,conf = pose

        conf_term = (conf > self.pcutoff)

        x_dist = abs(self.memory[-1, 1] - x)
        x_term = x_dist < self.maximum_distance

        y_dist = abs(self.memory[-1, 2] - y)
        y_term = y_dist < self.maximum_distance

        return conf_term and x_term and y_term


class KalmanFilter(object):
    def __init__(self):
        pass

class PoseFilter(Processor):
    def __init__(self,  **kwargs,):
        super().__init__(**kwargs)
        self.mode = kwargs['filter_type']

        if self.mode == "low_pass":
            self.min_cutoff = kwargs['minimum_cutoff']
            self.beta = kwargs['beta']
            self.point_filters = list()

        elif self.mode == "moving_average":
            self.pcutoff = kwargs['pcutoff']
            self.memory_length = kwargs['memory_length']
            self.maximum_distance = kwargs['maximum_distance']
            self.point_filters = list()

        elif self.mode == "kalman_filter":
            pass

        else:
            raise NotImplementedError("No such filtering algorithm has been implemented for the PoseFilter.")
    
    def reset(self):
        if self.mode == "low_pass":
            self.__init__(filter_type=self.mode, minimum_cutoff=self.min_cutoff, beta=self.beta)
        elif self.mode == "moving_average":
            self.__init__(filter_type=self.mode, pcutoff=self.pcutoff, maximum_distance=self.maximum_distance, memory_length=self.memory_length)

    def process(self, pose, **kwargs):
        time = kwargs['frame_time']

        if self.mode == "one_euro":
            for k, bp in enumerate(pose):
                try:
                    x_filter, y_filter = self.point_filters[k]
                    pose[k][0] = x_filter(time, bp[0])
                    pose[k][1] = y_filter(time, bp[1])
                # We hit this if there's no initialized filter.
                except IndexError:
                    self.point_filters.append((OneEuroFilter(time, bp[0], self.min_cutoff, self.beta), OneEuroFilter(time, bp[1], self.min_cutoff, self.beta)))
        elif self.mode == "moving_average":
            for k, bp in enumerate(pose):
                try:
                    pfilter = self.point_filters[k]
                    pose[k][0], pose[k][1], pose[k][2]  = pfilter(time, bp)
                # We hit this if there's no initialized filter.
                except IndexError:
                    self.point_filters.append(MovingAverageFilter(time, bp, pcutoff=self.pcutoff, maximum_distance=self.maximum_distance, memory_length=self.maximum_distance))

        
        return pose