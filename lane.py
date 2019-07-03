from line import Line
from config import Config
import numpy as np


class Lane:

    def __init__(self):

        # the number of current frame
        self.frame_number = 0

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        self.num_fail = Config.NUM_FAIL

        self.slope_diff = None

        self.slope_arctan_diff = None

        self.left_line = Line()

        self.right_line = Line()

    def get_line_base_pos(self):
        center = Config.IMG_SIZE[0] / 2

        lane_center = int(self.right_line.current_xfit[-1] - self.left_line.current_xfit[-1])

        self.line_base_pos = (center - lane_center) * Config.XM_PER_PIX

    def is_almost_parallel(self):
        slope_left = self.left_line.slope
        slope_right = self.right_line.slope

        slope_arctan_left = self.left_line.slope_arctan
        slope_arctan_right = self.right_line.slope_arctan

        self.slope_diff = abs(slope_right - slope_left)

        arctan_diff = slope_arctan_right - slope_arctan_left

        if arctan_diff > (np.pi - arctan_diff):
            arctan_diff = (np.pi - arctan_diff)

        self.slope_arctan_diff = arctan_diff

        if abs(self.slope_diff) < Config.SLOPE_DIFF_LIMIT:
            return True

        return False

    def sanity_check(self):

        if (abs(self.right_line.radius_of_curvature - self.left_line.radius_of_curvature) < Config.CURVATURE_DIFF_LIMIT or self.is_almost_parallel()) and\
                (Config.DISTANCE_BTW_LINES_LIMITS[0] < self.right_line.current_xfit[-1] - self.left_line.current_xfit[-1] < Config.DISTANCE_BTW_LINES_LIMITS[1]):
            return True

        return False

    def lane_info(self):
        return "{} | line_base_pos: {} | slope_diff: {} | slope_arctan_diff: {} \n\t LEFT: {} \n\t RIGHT: {}\n".format(
            self.frame_number, self.line_base_pos, self.slope_diff, self.slope_arctan_diff,
            self.left_line.line_info(), self.right_line.line_info())

    def print_frame_lane_info(self):
        return "Left line radius {0:.2f} | Right line radius {1:.2f} | Car center position {2:.2f}".format(
            self.left_line.radius_of_curvature, self.right_line.radius_of_curvature, self.line_base_pos)



