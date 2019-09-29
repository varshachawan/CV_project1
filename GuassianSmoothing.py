import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(image,kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))
    return output


def guass(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_blur(image, size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = guass(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()
    return convolution(image, kernel_2D)


if __name__ == '__main__':

    image = cv2.imread("Lena.png")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(image,cmap='gray')
    plt.title("Original Grey Image")
    plt.show()
    # the kernel size to k = {3, 5, 7, 11, 51} with fixed s = 1.
    for k in (3,5,7,11,51):
        output = gaussian_blur(image, k,1)
        plt.imshow(output, cmap='gray')
        plt.title("GuassianSmoothing for sigma = 1 and K = {}".format(k))
        plt.show()
    # the s = {0.1, 1, 2, 3, 5} with fixed kernel size k = 11.
    for s in (0.1, 1, 2, 3, 5):
        output = gaussian_blur(image, 11, s)
        plt.imshow(output, cmap='gray')
        plt.title("GuassianSmoothing for K = 11 and s = {}".format(s))
        plt.show()
