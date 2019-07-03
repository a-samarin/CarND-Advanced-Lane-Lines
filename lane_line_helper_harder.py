from lane import Lane
from line import Line
from config import Config
import numpy as np
import cv2


def points_extraction(images, nx, ny):
    """
    perform the extraction of object points and image points
    :param images: list of images
    :param nx: the number of inside corners in x
    :param ny: the number of inside corners in y
    :return: object points, image points and retvalues
    """

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    retlist = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        retlist.append(ret)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    #         # Draw and display the corners
    #         cv2.drawChessboardCorners(img, (8, 6), corners, ret)
    #         # write_name = 'corners_found'+str(idx)+'.jpg'
    #         # cv2.imwrite(write_name, img)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(500)
    #
    # cv2.destroyAllWindows()
    return objpoints, imgpoints, retlist


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = Config.NWINDOWS
    # Set the width of the windows +/- margin
    margin = Config.MARGIN
    # Set minimum number of pixels found to recenter window
    minpix = Config.MINPIX

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return leftx, lefty, rightx, righty, left_fitx, right_fitx, left_fit, right_fit


def fit_poly(line_x, line_y, plot_y):
    # Fit a second order polynomial to each with np.polyfit()
    line_fit = np.polyfit(line_y, line_x, 2)

    # Calc polynomial using ploty and line_fit
    line_fit_x = line_fit[0] * plot_y ** 2 + line_fit[1] * plot_y + line_fit[2]

    return line_fit_x, line_fit


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
    line_fit_x, line_fit = fit_poly(line_x, line_y, Config.PLOT_Y)

    return line_fit_x, line_x, line_y, line_fit


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


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, is_gray=True, orient='x', sobel_kernel=3, sob_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale

    gray = np.copy(img)
    if not is_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= sob_thresh[0]) & (scaled_sobel <= sob_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, is_gray=True, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = np.copy(img)
    if is_gray == False:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image

    return binary_output


def color_gradient_pipline(img, h_thresh=(19, 90), sx_thresh=(15, 110), r_thresh=(225, 255)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    h_channel = hls[:, :, 0]

    # h_thresh = (19, 90)
    h_binary_output = np.zeros_like(h_channel)
    h_binary_output[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # R threshold
    R = img[:, :, 2]
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1

    # Sobel x
    r_sx_binary = abs_sobel_thresh(R, orient='x', sobel_kernel=5, sob_thresh=sx_thresh)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(h_binary_output)
    combined_binary[(h_binary_output == 1) | (r_sx_binary == 1) | (r_binary == 1)] = 1

    return combined_binary


def color_gradient_pipline_challenge(img, s_thresh=(30, 140), r_thresh = (182, 255), b_thresh = (182, 255), s_mag_thresh = (60, 110)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    #h_channel = hls[:,:,0]
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))

    # dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

#     h_thresh = (10, 70)
#     h_binary_output = np.zeros_like(h_channel)
#     h_binary_output[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    #s_thresh = (75, 140)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # R threshold
    R = img[:,:,2]
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1

    B = img[:,:,0]
    b_binary = np.zeros_like(B)
    b_binary[(B > b_thresh[0]) & (B <= b_thresh[1])] = 1

    # Magnitude
    s_mag_binary = mag_thresh(s_channel, sobel_kernel=13, mag_thresh=s_mag_thresh)

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_mag_binary == 1) | (r_binary == 1) | (s_binary == 1) | (b_binary == 1)] = 1

    return combined_binary


def draw_track_with_lines(undist, warped, left_fitx, right_fitx, ploty, lane_info):

    margin = Config.LINE_WIDTH

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # bwn_window_img = np.zeros_like(out_img)
    left_lane_line_window = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    right_lane_line_window = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    lane_line_pts = np.hstack((left_lane_line_window, right_lane_line_window))

    cv2.fillPoly(color_warp, np.int_([lane_line_pts]), (255, 0, 0))

    newwarp = cv2.warpPerspective(color_warp, Config.M_INV, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Write lane info
    print_lane_info(result, lane_info)

    return result


def print_lane_info(img, lane_info):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    y0, dy = 40, 40
    for i, line in enumerate(lane_info.split('|')):
        y = y0 + i * dy
        cv2.putText(img, line.strip(),
                    (50, y),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

