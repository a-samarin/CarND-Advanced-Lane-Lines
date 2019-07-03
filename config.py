class Config:
    # the number of iteration that we consider as "the last n iterations"
    NUM_ITER = 15

    # the number of failer iteratioin
    NUM_FAIL = 4

    # meters per pixel in y dimension
    YM_PER_PIX = 30 / 720

    # meters per pixel in x dimension
    XM_PER_PIX = 3.7 / 650

    # H channel threshold
    H_THRESH = (19, 90)

    # R channel threshold
    R_THRESH = (225, 255)

    # Soble x threshold
    SX_THRESH = (15, 110)

    # (50, 140)
    S_THRESH_CH = (30, 140)

    # (182, 255)
    R_THRESH_CH = (182, 255)

    #  (182, 255)
    B_THRESH_CH = (182, 255)

    # (65, 110)
    S_MAG_THRESH_CH = (60, 110)

    IMG_SIZE = None

    PLOT_Y = None

    Y_EVAL = None

    Y_EVAL2 = None

    # the number of sliding windows
    NWINDOWS = 9

    # the width of the margin around the previous polynomial to search
    # 100
    MARGIN = 45

    # minimum number of pixels found to recenter window
    # 50
    MINPIX = 65

    # width of line that shows on image
    LINE_WIDTH = 25

    # camera matrix
    MTX = None

    # distortion coefficients
    DIST = None

    # the perspective transform matrix
    M = None

    # the 'invert' perspective transform matrix
    M_INV = None

    # indicates the valuse between which radius curvature can be
    # RADIUS_CURVATURE_LIMITS = (250.0, 1250.0)

    # max difference between lines' slopes
    SLOPE_DIFF_LIMIT = 0.19

    # max difference between lines' curvatures
    CURVATURE_DIFF_LIMIT = 600.0

    # the 'window' of approximately right distance horizontally between two lines
    DISTANCE_BTW_LINES_LIMITS = (480, 700)


class ConfigHarder:
    # the number of iteration that we consider as "the last n iterations"
    NUM_ITER = 15

    # the number of failer iteratioin
    NUM_FAIL = 4

    # meters per pixel in y dimension
    YM_PER_PIX = 30 / 720

    # meters per pixel in x dimension
    XM_PER_PIX = 3.7 / 650

    # H channel threshold
    H_THRESH = (19, 90)

    # R channel threshold
    R_THRESH = (225, 255)

    # Soble x threshold
    SX_THRESH = (15, 110)

    # (50, 140)
    S_THRESH_CH = (30, 140)

    # (182, 255)
    R_THRESH_CH = (182, 255)

    #  (182, 255)
    B_THRESH_CH = (182, 255)

    # (65, 110)
    S_MAG_THRESH_CH = (60, 110)

    IMG_SIZE = None

    PLOT_Y = None

    Y_EVAL = None

    Y_EVAL2 = None

    # the number of sliding windows
    NWINDOWS = 9

    # the width of the margin around the previous polynomial to search
    # 100
    MARGIN = 45

    # minimum number of pixels found to recenter window
    # 50
    MINPIX = 65

    # width of line that shows on image
    LINE_WIDTH = 25

    # camera matrix
    MTX = None

    # distortion coefficients
    DIST = None

    # the perspective transform matrix
    M = None

    # the 'invert' perspective transform matrix
    M_INV = None

    # indicates the valuse between which radius curvature can be
    # RADIUS_CURVATURE_LIMITS = (250.0, 1250.0)

    # max difference between lines' slopes
    SLOPE_DIFF_LIMIT = 0.19

    # max difference between lines' curvatures
    CURVATURE_DIFF_LIMIT = 600.0

    # the 'window' of approximately right distance horizontally between two lines
    DISTANCE_BTW_LINES_LIMITS = (480, 700)