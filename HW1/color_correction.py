import cv2
import numpy as np
import os


"""
TODO White patch algorithm
"""
def white_patch_algorithm():
    raise NotImplementedError


"""
TODO Gray-world algorithm
"""
def gray_world_algorithm():
    raise NotImplementedError


"""
Bonus 
"""
def other_white_balance_algorithm():
    raise NotImplementedError


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

if __name__ == "__main__":
    main()