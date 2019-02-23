from lane import Lane
from line import Line
from config import Config
import numpy as np
import cv2


def fit_poly(img_shape, line_x, line_y):
    # Fit a second order polynomial to each with np.polyfit()
    line_fit = np.polyfit(line_y, line_x, 2)

    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calc polynomial using ploty and line_fit
    line_fit_x = line_fit[0] * ploty ** 2 + line_fit[1] * ploty + line_fit[2]

    return line_fit_x, ploty


def search_around_poly(binary_warped, line_fit):

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    line_inds = (
            (nonzerox > (line_fit[0] * (nonzeroy ** 2) + line_fit[1] * nonzeroy + line_fit[2] - Config.MARGIN))
            & (nonzerox < (line_fit[0] * (nonzeroy ** 2) + line_fit[1] * nonzeroy + line_fit[2] + Config.MARGIN))
    )

    # extract line pixel positions
    line_x = nonzerox[line_inds]
    line_y = nonzeroy[line_inds]

    # Fit new polynomials
    line_fit_x, ploty = fit_poly(binary_warped.shape, line_x, line_y)

    return line_fit_x


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image
