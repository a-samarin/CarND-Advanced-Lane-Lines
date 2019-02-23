import numpy as np
from config import Config


# Define a class to receive the characteristics of each line detection
class Line:

    def __init__(self):

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        self.current_xfit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

    def line_info(self):
        return "/n/t {} | {}".format(self.detected, self.radius_of_curvature)

    def line_check(self):
        pass

    def get_radius_of_curvature(self):
        self.radius_of_curvature = np.polyfit(self.ally * Config.YM_PER_PIX, self.allx * Config.XM_PER_PIX, 2)
