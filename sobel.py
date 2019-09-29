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

    return output

if __name__ == "__main__":

    image = cv2.imread("Lena.png")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(image,cmap='gray')
    plt.title("Original Grey Image")
    plt.show()

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gx = filter
    Gy = np.flip(filter.T, axis=0)

    new_image_x = convolution(image, Gx)

    plt.imshow(new_image_x, cmap='gray')
    plt.title("Horizontal Edge")
    plt.show()

    new_image_y = convolution(image,Gy )


    plt.imshow(new_image_y, cmap='gray')
    plt.title("Vertical Edge")
    plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    # Normalisation

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Gradient combined Gx,GY ")
    plt.show()



