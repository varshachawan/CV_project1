import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.float32)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('Lena.png',0) # Only for grayscale image
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

noise_img = sp_noise(image,0.2)
plt.imshow(noise_img, cmap='gray')
plt.title("Median Filtering on twice upsampled image")
plt.show()
