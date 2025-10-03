import numpy as np
import cv2
import argparse
import signal

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
    k_height, k_width = kernel.shape
    i_height, i_width = input_img.shape
    pad_height, pad_width = k_height // 2, k_width // 2
    
    padded_image = np.pad(input_img, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
    output_img = np.zeros_like(input_img, dtype=np.float64)
    
    for y in range(i_height):
        for x in range(i_width):
            roi = padded_image[y : y + k_height, x : x + k_width]
            output_img[y, x] = (roi * kernel).sum()
    output_img = output_img.astype(np.uint8)  
    ############### YOUR CODE ENDS HERE #################
    return output_img
    
def gaussian_filter(input_img, kernel_size, sigma):
    ############### YOUR CODE STARTS HERE ###############
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    center = kernel_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    gaussian_kernel = np.exp(-(x**2+y**2) / (2 * sigma**2))

    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, gaussian_kernel)

def median_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    noise = np.zeros_like(gray_img)
    cv2.randu(noise, 0, 255)
    noisy_image = gray_img.copy()
    noisy_image[noise < 10] = 0   # pepper
    noisy_image[noise > 245] = 255 # salt

    k_height, k_width = kernel.shape
    i_height, i_width = input_img.shape
    pad_height, pad_width = k_height // 2, k_width // 2
    
    padded_image = np.pad(input_img, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
    output_img = np.zeros_like(input_img, dtype=np.float64)
    
    for y in range(i_height):
        for x in range(i_width):
            roi = padded_image[y : y + k_height, x : x + k_width]
            output_img[y, x] = (roi * kernel).sum()
    output_img = output_img.astype(np.uint8)
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
        kernel_sizes = [3, 7, 11]
        sigmas = [1.0, 2.0, 3.0]
        for kernel_size in kernel_sizes:
            for sigma in sigmas:
                output_img = gaussian_filter(input_img, kernel_size, sigma)
                cv2.imwrite(f"output_gaussian_{kernel_size}_{sigma}.jpg", output_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
        cv2.imwrite("output_median.jpg", output_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
        cv2.imwrite("output_laplacian.jpg", output_img)

    