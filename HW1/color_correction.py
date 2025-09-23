import cv2
import numpy as np
import os


"""
TODO White patch algorithm
"""
def white_patch_algorithm(img):
    """
    white_patch_algorithm for color correction

    Input:
        img (np.array): A 2D NumPy array

    output:
        np.array: a color corrected image
    """
    b, g, r = cv2.split(img)
    
    percentile = 99
    b_max = np.percentile(b, percentile)
    g_max = np.percentile(g, percentile)
    r_max = np.percentile(r, percentile)
    # g_max = g.max()
    # r_max = r.max()
    # b_max = b.max()

    b_gain = 255.0 / b_max if b_max > 0 else 1 # avoid to divide 0
    g_gain = 255.0 / g_max if g_max > 0 else 1
    r_gain = 255.0 / r_max if r_max > 0 else 1
    
    b_corrected = np.clip(b.astype(np.float64) * b_gain, 0, 255)
    g_corrected = np.clip(g.astype(np.float64) * g_gain, 0, 255)
    r_corrected = np.clip(r.astype(np.float64) * r_gain, 0, 255)

    corrected_img = cv2.merge((b_corrected, g_corrected, r_corrected)).astype(np.uint8)

    return corrected_img
"""
TODO Gray-world algorithm
"""
def gray_world_algorithm(img):
    """
    gray_world_algorithm for color correction

    Input:
        img (np.array): A 2D NumPy array

    output:
        np.array: a color corrected image
    """
    b, g, r = cv2.split(img)

    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    gray_avg = (b_avg + g_avg + r_avg) / 3.0

    b_gain = gray_avg / b_avg if b_avg > 0 else 1 # avoid to divide 0
    g_gain = gray_avg / g_avg if g_avg > 0 else 1
    r_gain = gray_avg / r_avg if r_avg > 0 else 1

    b_corrected = np.clip(b.astype(np.float64) * b_gain, 0, 255)
    g_corrected = np.clip(g.astype(np.float64) * g_gain, 0, 255)
    r_corrected = np.clip(r.astype(np.float64) * r_gain, 0, 255)

    corrected_img = cv2.merge((b_corrected, g_corrected, r_corrected)).astype(np.uint8)

    return corrected_img

"""
Bonus 
"""
def other_white_balance_algorithm(img, white_patch_roi):
    """
    White Point Correction
    Corrects the color cast of an image using a reference white patch.

    Input:
        img (np.array): A 2D NumPy array image need to be corrected.
        white_patch_roi (tuple): A tuple of slices defining the white patch
                                 (y_start:y_end, x_start:x_end).

    Output:
        np.array: The color-corrected BGR image.
    """
    y_start, y_end = white_patch_roi[0].start, white_patch_roi[0].stop
    x_start, x_end = white_patch_roi[1].start, white_patch_roi[1].stop
    patch = img[y_start:y_end, x_start:x_end]

    b_avg, g_avg, r_avg = patch.mean(axis=(0,1))

    gray_avg = (b_avg + g_avg + r_avg) / 3

    b_gain = gray_avg / b_avg if b_avg > 0 else 1 # avoid to divide 0
    g_gain = gray_avg / g_avg if g_avg > 0 else 1
    r_gain = gray_avg / r_avg if r_avg > 0 else 1

    b, g, r = cv2.split(img)
    b_corrected = np.clip(b.astype(np.float64) * b_gain, 0, 255)
    g_corrected = np.clip(g.astype(np.float64) * g_gain, 0, 255)
    r_corrected = np.clip(r.astype(np.float64) * r_gain, 0, 255)

    corrected_img = cv2.merge((b_corrected, g_corrected, r_corrected)).astype(np.uint8)

    return corrected_img


"""
Main function
"""
def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))

        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)

        cv2.imwrite("result/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("result/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)
    
    img = cv2.imread("data/color_correction/input1.bmp")
    roi_input1 = (slice(411,458), slice(201,248))
    white_point_img = other_white_balance_algorithm(img, roi_input1)
    cv2.imwrite("result/color_correction/white_point_input1.bmp", white_point_img)

    img = cv2.imread("data/color_correction/input2.bmp")
    roi_input2 = (slice(304,311), slice(93,106))
    white_point_img = other_white_balance_algorithm(img, roi_input2)
    cv2.imwrite("result/color_correction/white_point_input2.bmp", white_point_img)

if __name__ == "__main__":
    main()