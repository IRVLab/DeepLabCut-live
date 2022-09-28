""""
IRVLab GPLV3

A low pass filter for pose estimators.

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
    def __init__(self, minimum_cutoff=0.005, beta=0.7, confidence_thresh=0.8, memory_length=5, max_center_dist=500, max_self_dist=50, **kwargs,):
        super().__init__(**kwargs)
        self.min_cutoff = minimum_cutoff
        self.beta = beta
        self.point_filters = list()

        self.confidence_thresh= confidence_thresh

        self.memory_length = memory_length
        self.past_poses = [None] * memory_length
        self.past_pose_centers = [None] * memory_length
        self.past_pose_times = [None] * memory_length
        self.max_center_dist = max_center_dist
        self.max_self_dist = max_self_dist
    
    def reset(self):
        self.__init__()

    def calculate_pose_center(self, pose):
        x_sum = np.sum(pose[:,0])
        y_sum = np.sum(pose[:,1])
        centroid = (int(x_sum/len(pose)), int(y_sum/len(pose)))
        return centroid

    def check_pose_dist(self, c, bp):
        distance = math.sqrt((c[0] - bp[0])**2 + (c[1] - bp[1])**2 )
        return (distance < self.max_center_dist)
        
    def check_prev_dist(self, k, bp, current_time):
        pose = self.past_poses[-1]
        if pose is None:
            return True

        # last_time = self.past_pose_times[-1]
        distance = math.sqrt((pose[k][0] - bp[0])**2 + (pose[k][1] - bp[1])**2 )
        
        return (distance < self.max_self_dist)

    def average_past_point(self, k):
        x_avg, y_avg, w_sum, n = [0,0,0,0]
        for pose in self.past_poses:
            if pose is not None: #Check for None
                x_avg += pose[k][0] * (pose[k][2] * 100)
                y_avg += pose[k][1] * (pose[k][2] * 100)
                w_sum += pose[k][2] * 100
                n += 1
        
        if n == self.memory_length:
            x_avg = x_avg/w_sum
            y_avg = y_avg/w_sum
            conf_avg = (w_sum/n)/100

            return (int(x_avg),int(y_avg), conf_avg)

        else:
            return None

    def process(self, pose, **kwargs):
        time = kwargs['frame_time']

        # First, run the points through a point-by-point filtering process to smooth them out a little bit.
        for k, bp in enumerate(pose):
            try:
                x_filter, y_filter = self.point_filters[k]
                pose[k][0] = x_filter(time, bp[0])
                pose[k][1] = y_filter(time, bp[1])
            # We hit this if there's no initialized filter.
            except IndexError:
                self.point_filters.append((OneEuroFilter(time, bp[0], self.min_cutoff, self.beta), OneEuroFilter(time, bp[1], self.min_cutoff, self.beta)))

        
        # Next, let's make sure that the points are all within a reasonable range from the pose center (which we'll also calculate here.)
        # If a point is not a reasonable distance from the pose center, we'll generate a point based on our memory of the last N points.
        pose_center = self.calculate_pose_center(pose)
        for k, bp in enumerate(pose):
            if self.check_pose_dist(pose_center,bp):
                pass
            elif self.check_prev_dist(k, bp, time):
                pass
            elif bp[2] >= self.confidence_thresh:
                pass
            else:
                average_of_past = self.average_past_point(k)
                if average_of_past is not None:
                    pose[k][0] = average_of_past[0]
                    pose[k][1] = average_of_past[1]
                    pose[k][2] = average_of_past[2]
                else:
                    pass # There's nothing we can do if there's not enough past data.

        #Lastly, update memory. 
        self.past_poses.pop(0)
        self.past_poses.append(pose)

        self.past_pose_times.pop(0)
        self.past_pose_times.append(time)

        self.past_pose_centers.pop(0)
        self.past_pose_centers.append(pose_center)

        return pose
