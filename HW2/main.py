import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############

    ############### YOUR CODE ENDS HERE #################
    return output_img

def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############

    ############### YOUR CODE ENDS HERE #################
    return output_img
    
def gaussian_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############

    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def median_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############

    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img):
    ############### YOUR CODE STARTS HERE ###############

    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)

    cv2.imwrite("output.jpg", output_img)