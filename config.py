class Config:
    # the number of iteration that we consider as "the last n iterations"
    NUM_ITER = 30

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

    # the width of the margin around the previous polynomial to search
    MARGIN = 100

    # indicates the valuse between which radius curvature can be
    # RADIUS_CURVATURE_LIMITS = (250.0, 1250.0)

    # max difference between line curvatures
    CURVATURE_DIFF_LIMIT = 250.0

    # the 'window' of approximately right distance horizontally between two lines
    DISTANCE_BTW_LINES_LIMITS = (600, 800)
