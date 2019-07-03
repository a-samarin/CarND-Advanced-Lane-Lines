import numpy as np
from config import Config


# Define a class to receive the characteristics of each line detection
class Line:

    def __init__(self):

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = None

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients of the last n fits of the line
        self.recent_fitted = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        self.current_xfit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        self.slope = None

        self.slope_arctan = None

    def _update_radius_of_curvature(self):
        fit_cr = np.polyfit(self.ally * Config.YM_PER_PIX, self.allx * Config.XM_PER_PIX, 2)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * Config.Y_EVAL * Config.YM_PER_PIX + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    def _update_xfit(self):
        # self.current_xfit = current_xfit

        if self.recent_xfitted is None:
            self.recent_xfitted = [self.current_xfit] * Config.NUM_ITER
        else:

            # self.recent_xfitted = self.recent_xfitted[1:]
            # self.recent_xfitted = np.concatenate(self.recent_xfitted, self.current_xfit)

            self.recent_xfitted.pop(0)
            self.recent_xfitted.append(self.current_xfit)
            # self.recent_xfitted.append(self.current_xfit, axis=0)

        self.bestx = np.mean(self.recent_xfitted, axis=0)

    def _update_fit(self):
        # self.current_fit = current_fit

        if self.recent_fitted is None:
            self.recent_fitted = [self.current_fit] * Config.NUM_ITER
        else:

            # self.recent_fitted = self.recent_fitted[1:]
            # self.recent_fitted = np.concatenate(self.recent_fitted, self.current_fit)

            self.recent_fitted.pop(0)
            self.recent_fitted.append(self.current_fit)
            # self.recent_fitted.append(self.current_fit, axis=0)
        self.best_fit = np.mean(self.recent_fitted, axis=0)

    def _update_slopes(self):
        y1 = Config.Y_EVAL2
        y2 = Config.Y_EVAL

        x1 = self.current_fit[0] * y1 ** 2 + self.current_fit[1] * y1 + self.current_fit[2]
        x2 = self.current_fit[0] * y2 ** 2 + self.current_fit[1] * y2 + self.current_fit[2]

        self.slope = (x2 - x1)/(y2 - y1)
        self.slope_arctan = np.arctan(self.slope)

    def line_info(self):
        return "radius_of_curvature: {} | slope: {} | current_fit: {}".format(self.radius_of_curvature, self.slope, self.current_fit)

    def refresh_line(self):
        self.recent_xfitted = None
        self.recent_fitted = None

    def line_check(self):
        pass

    def update_current(self, allx, ally, current_xfit, current_fit):
        self.allx = allx
        self.ally = ally
        self.current_xfit = current_xfit
        self.current_fit = current_fit

        self._update_radius_of_curvature()
        self._update_slopes()

    def update_line_fit(self):
        self._update_xfit()
        self._update_fit()
        # self._update_radius_of_curvature(y_eval)

