from line import Line
from config import Config
import numpy as np


class Lane:

    def __init__(self, img_size):

        # the number of current frame
        self.frame_number = 0

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        self.img_size = img_size

        self.left_line = Line()

        self.right_line = Line()

    def get_line_base_pos(self):
        center = self.img_size[0] / 2

        lane_center = int(self.right_line.current_xfit[-1] - self.left_line.current_xfit[-1])

        self.line_base_pos = (center - lane_center) * Config.XM_PER_PIX

    def sanity_check(self):

        if (abs(self.right_line.radius_of_curvature - self.left_line.radius_of_curvature) < Config.CURVATURE_DIFF_LIMIT) and\
                (Config.DISTANCE_BTW_LINES_LIMITS[0] < self.right_line.current_xfit[-1] - self.left_line.current_xfit[-1] < Config.DISTANCE_BTW_LINES_LIMITS[1]):
            return True

        return False

    def lane_info(self):
        return "{} | {} {} {}".format(self.frame_number, self.line_base_pos,
                                      self.left_line.line_info(), self.right_line.line_info())

    def print_frame_lane_info(self):
        return "Left line radius {} \n Right line radius {} \n Car center position {}".format(
            self.left_line.radius_of_curvature, self.right_line.radius_of_curvature, self.line_base_pos)
