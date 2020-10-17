import cv2
import numpy as np


def show_weighted(image0, alpha0, image1, alpha1):
    #image0 = np.array(image0, dtype = np.uint8)
    #image1 = np.array(image1, dtype = np.uint8)
    out = cv2.addWeighted(image0, alpha0, image1, alpha1, 0.3)
    return out