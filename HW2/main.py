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

def padding(input_img, kernel_height, kernel_width):
    ############### YOUR CODE STARTS HERE ###############
    i_height, i_width = input_img.shape
    pad_height, pad_witdh = kernel_height // 2, kernel_width // 2

    new_height, new_width = i_height + 2 * pad_height, i_width + 2 * pad_witdh
    output_img = np.zeros((new_height, new_width), dtype=input_img.dtype)

    output_img[pad_height : pad_height + i_height, pad_witdh : pad_witdh + i_width] = input_img
    ############### YOUR CODE ENDS HERE #################
    return output_img

def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    k_height, k_width = kernel.shape
    i_height, i_width = input_img.shape

    padded_img = padding(input_img, k_height, k_width)
    output_img = np.zeros_like(input_img, dtype=np.float64)
    
    for y in range(i_height):
        for x in range(i_width):
            roi = padded_img[y : y + k_height, x : x + k_width]
            output_img[y, x] = (roi * kernel).sum()
    ############### YOUR CODE ENDS HERE #################
    return output_img
    
def gaussian_filter(input_img, kernel_size, sigma):
    ############### YOUR CODE STARTS HERE ###############
    b, g, r = cv2.split(input_img)

    center = kernel_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    gaussian_kernel = np.exp(-(x**2+y**2) / (2 * sigma**2))

    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    b_filter = convolution(b, gaussian_kernel)
    g_filter = convolution(g, gaussian_kernel)
    r_filter = convolution(r, gaussian_kernel)

    output_img = cv2.merge((b_filter, g_filter, r_filter))
    output_img = np.clip(output_img, 0, 255)
    ############### YOUR CODE ENDS HERE #################
    return output_img.astype(np.uint8)  

def median_filter(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    noise = np.zeros_like(gray_img)
    cv2.randu(noise, 0, 255)
    noisy_image = gray_img.copy()
    noisy_image[noise < 10] = 0   # pepper
    noisy_image[noise > 245] = 255 # salt

    i_height, i_width = noisy_image.shape
    
    padded_image = padding(noisy_image, kernel_size, kernel_size)
    output_img = np.zeros_like(input_img, dtype=np.float64)
    
    for y in range(i_height):
        for x in range(i_width):
            roi = padded_image[y : y + kernel_size, x : x + kernel_size]
            median_value = np.median(roi)
            output_img[y, x] = median_value
    output_img = output_img.astype(np.uint8)
    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    b, g, r = cv2.split(input_img)

    b_filter = convolution(b, kernel)
    g_filter = convolution(g, kernel)
    r_filter = convolution(r, kernel)

    output_img = cv2.merge((b_filter, g_filter, r_filter))
    output_img = np.clip(output_img, 0, 255)
    ############### YOUR CODE ENDS HERE #################
    return output_img.astype(np.uint8)

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
        kernel_sizes = [3, 7, 11]
        for kernel_size in kernel_sizes:
            output_img = median_filter(input_img, kernel_size)
            cv2.imwrite(f"output_median_{kernel_size}.jpg", output_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        kernel_list = [
            [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]],
            
            [[-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]]
        ]
        kernel_np = np.array(kernel_list)
        for i, kernel in enumerate(kernel_np):
            output_img = laplacian_sharpening(input_img, kernel)
            cv2.imwrite(f"output_laplacian_{i+1}.jpg", output_img)

    