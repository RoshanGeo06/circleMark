import cv2 as cv
import numpy as np
import sys

def main(argv):
    default_file = '1.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    # Apply median blur to the image
    src = cv.medianBlur(src, 3)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection to the image
    edges = cv.Canny(gray, 50, 150)

    gray_blurred = cv.blur(edges, (3, 3))
    rows = gray.shape[0]

    # Detect larger circles
    detected_circles_large = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, rows / 8,
                                             param1=50, param2=30, minRadius=15, maxRadius=30)

    # Mask the larger circles
    if detected_circles_large is not None:
        mask = np.zeros_like(gray)
        detected_circles_large = np.uint16(np.around(detected_circles_large))
        for i in detected_circles_large[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(mask, center, radius, (255), thickness=-1)

        # Invert the mask
        mask = cv.bitwise_not(mask)
        # Bitwise-AND mask and original image
        gray_blurred = cv.bitwise_and(gray_blurred, gray_blurred, mask=mask)

    # Detect smaller circles
    detected_circles_small = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, rows / 8,
                                             param1=50, param2=30, minRadius=1, maxRadius=14)

    # Draw the circles
    if detected_circles_large is not None:
        for i in detected_circles_large[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    if detected_circles_small is not None:
        detected_circles_small = np.uint16(np.around(detected_circles_small))
        for i in detected_circles_small[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", src)
    cv.waitKey(0)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
