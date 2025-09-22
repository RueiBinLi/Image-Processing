import cv2
import numpy as np
import os


"""
TODO Binary transfer
"""
def to_binary():
    raise NotImplementedError


"""
TODO Two-pass algorithm
"""
def two_pass():
    raise NotImplementedError


"""
TODO Seed filling algorithm
"""
def seed_filling():
    raise NotImplementedError


"""
Bonus
"""
def other_cca_algorithm():
    raise NotImplementedError


"""
TODO Color mapping
"""
def color_mapping():
    raise NotImplementedError


"""
Main function
"""
def main():

    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img)

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
        
            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite("result/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)


if __name__ == "__main__":
    main()