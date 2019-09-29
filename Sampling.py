import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


def UpSampled(image):
    p=[]
    pixels= np.array(image)
    pixels = pixels.astype("float32")
    for i in range(1,len(pixels)+1,1):
        p.append(i)
    pixels = np.insert(pixels,p,0,axis=0)
    pixels = np.insert(pixels,p,0,axis=1)
    img_up= Image.fromarray(pixels)
    return img_up


def upSampled_normal(n,image):
    w, h = n* image.shape[0], n * image.shape[1]
    im_up = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            im_up[i, j] = image[i // n, j // n]
    return im_up


def downSample(n,image):
    w, h = image.shape[0] // n, image.shape[1] // n
    im_small = np.zeros((w,h))
    for i in range(w):
       for j in range(h):
          im_small[i,j] = image[n*i, n*j]
    return im_small


if __name__ == "__main__":
    image1 = cv2.imread("Lena.png")
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # to check the intensity of pixels

    # rows, cols = image.shape
    #
    # for i in range(rows):
    #     for j in range(cols):
    #         k = img[i, j]
    #         print(k)

    print("Shape of gray scale image before sampling",image.shape)
    plt.imshow(image, cmap='gray')
    plt.title("Input Gray Scale Image")
    plt.show()

    # Down sampling once
    Down1_img = downSample(2,image)
    print("Shape after downsampling first time",Down1_img.shape)
    plt.imshow(Down1_img,cmap='gray')
    plt.title("Downsampled fist time to Half Value")
    plt.show()

    # Down Sampling twice
    Down2_img = downSample(2, Down1_img)
    print("Shape after downsampling Second time",Down2_img.shape)
    plt.imshow(Down2_img, cmap='gray')
    plt.title("Downsampled twice")
    plt.show()

    # Upsampling by Normal method

    N_up1_img = upSampled_normal(2,Down2_img)
    # # im_up = cv2.resize(im_small_2,(256,256))
    print("size of image upsampled once Normal Method",N_up1_img.shape)
    plt.imshow(N_up1_img, cmap='gray')
    plt.title("Upsampled first time by normal method")
    plt.show()

    N_up2_img = upSampled_normal(2, N_up1_img)
    print("size of image upsampled twice by Normal Method", N_up2_img.shape)
    plt.imshow(N_up2_img, cmap='gray')
    plt.title("upsampled second time by normal method")
    plt.show()

    # Upsampling by inserting empty pixels

    up1_img = UpSampled(Down2_img)
    print("size of image upsampled once by inserting empty pixel", up1_img.size)
    plt.imshow(up1_img, cmap='gray')
    plt.title("Upsampled first time by inserting empty pixel")
    plt.show()

    up2_img = UpSampled(up1_img)
    print("size of image upsampled twice by inserting empty pixel", up2_img.size)
    plt.imshow(up2_img, cmap='gray')
    plt.title("Upsampled second time by inserting empty pixel")
    plt.show()

