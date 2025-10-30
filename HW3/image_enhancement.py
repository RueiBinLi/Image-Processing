import cv2
import numpy as np


"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img, c, gamma):
    table = np.array([
        ((c * (i / 255.0)) ** gamma) * 255.0
        for i in np.arange(0, 256)
    ])

    table = np.clip(table, 0, 255).astype("uint8")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2]
    new_v = table[v_channel]
    corrected_hsv = hsv_img.copy()
    corrected_hsv[:, :, 2] = new_v
    corrected_img = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

    return corrected_img


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2]
    hist, bins = np.histogram(v_channel, 256, [0, 256])
    
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked, 0).astype("uint8")

    v_equalized = cdf_final[v_channel]

    corrected_hsv = hsv_img.copy()
    corrected_hsv[:, :, 2] = v_equalized
    corrected_img = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

    return corrected_img

"""
Bonus
"""
def other_enhancement_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [0.6, 1, 3] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, 1, gamma)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        output_dir = "./gamma_" + str(gamma) + ".bmp"
        cv2.imwrite(output_dir, gamma_correction_img)
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    output_dir = "./histogram.bmp"
    cv2.imwrite(output_dir, histogram_equalization_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
