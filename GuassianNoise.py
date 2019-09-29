import matplotlib.pyplot as plt
import cv2
import numpy as np
import GuassianSmoothing as G
import medianFilter as M
from PIL import Image
import random


image = cv2.imread("Lena.png")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(image, cmap = 'gray')
plt.title("original grey image")
plt.show()

gauss = np.random.normal(0 ,0.1 ,image.shape)
noisy_image = image + gauss
print("noisy image shape",noisy_image.shape)
plt.imshow(noisy_image, cmap='gray')
plt.title("Guassian noisy image")
plt.show()

Guass_filtered = G.gaussian_blur(noisy_image, 11, 1)
plt.imshow(Guass_filtered, cmap='gray')
plt.title("Guass smoothing on guass noise")
plt.show()

arr = np.array(noisy_image)
Medain_filtered = M.median_filter(arr,3)
img = Image.fromarray(Medain_filtered)
plt.imshow(img, cmap='gray')
plt.title("median smoothing on guass noise")
plt.show()

prob = 0.2
output = np.zeros(image.shape, np.float32)
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

plt.imshow(output, cmap='gray')
plt.title("SP noise")
plt.show()
GuassFilteredSP = G.gaussian_blur(output,11,1)
plt.imshow(GuassFilteredSP, cmap='gray')
plt.title("Guass smoothing on sp noise")
plt.show()

arr1 = np.array(output)
Medain_filtered_sp = M.median_filter(arr1,7)
img1 = Image.fromarray(Medain_filtered_sp)
plt.imshow(img1, cmap='gray')
plt.title("median smoothing on sp noise")
plt.show()


