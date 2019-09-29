import matplotlib.pyplot as plt
import cv2
import numpy as np
import GuassianF as G
import medianFilter as M
import Sampling as S
from PIL import Image

if __name__ == "__main__":
    image1 = cv2.imread("Lena.png")
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print("Shape of gray scale image before sampling",image.shape)
    plt.imshow(image, cmap='gray')
    plt.title("Original Gray Scale Image")
    plt.show()

    D1 = S.downSample(2,image)
    D2 = S.downSample(2,D1)
    # plt.imshow(D2, cmap='gray')
    # plt.title("D2")
    # plt.show()
    U1 = S.UpSampled(D2)
    plt.imshow(U1, cmap='gray')
    plt.title("UP sampled once by inserting empty pixel")
    plt.show()

    U1_G = G.gaussian_blur(U1,11,1)
    plt.imshow(U1_G, cmap='gray')
    plt.title("Guassian smoothing on one time upsampled image")
    plt.show()

    U2= S.UpSampled(U1)
    plt.imshow(U2, cmap='gray')
    plt.title("UP sampled twice by inserting empty pixel")
    plt.show()

    U2_G = G.gaussian_blur(U2, 11, 1)
    plt.imshow(U2_G, cmap='gray')
    plt.title("Guassian smoothing on second time upsampled image")
    plt.show()

    # by sending up sampled image
    U2_2 = S.UpSampled(U1_G)
    plt.imshow(U2_2, cmap='gray')
    plt.title("UP sampled twice -input Guass smooth image")
    plt.show()

    U2_G_2 = G.gaussian_blur(U2_2, 11, 1)
    plt.imshow(U2_G_2, cmap='gray')
    plt.title("Guassian smootheing on twice UPsampled")
    plt.show()

    arr = np.array(U1)
    U1_M_arr = M.median_filter(arr,3)
    U1_M = Image.fromarray(U1_M_arr)
    plt.imshow(U1_M, cmap='gray')
    plt.title("Median Filtering on one time upsampled image")
    plt.show()

    arr2 = np.array(U2)
    U2_M_arr = M.median_filter(arr2, 3)
    U2_M = Image.fromarray(U2_M_arr)
    plt.imshow(U2_M, cmap='gray')
    plt.title("Median Filtering on twice upsampled image")
    plt.show()

    # # by sending up sampled image
    # U2_2 = S.UpSampled(U1_M)
    # plt.imshow(U2_2, cmap='gray')
    # plt.title("UP sampled twice -input med smooth image")
    # plt.show()
    # arr3 = np.array(U2_2)
    # U2_M_2 = M.median_filter(arr3,3)
    # plt.imshow(U2_M_2, cmap='gray')
    # plt.title("med smootheing on twice UPsampled")
    # plt.show()



