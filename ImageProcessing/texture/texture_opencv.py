#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import os
#import sys
#from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2


class texture():

    def __init__(self, ps = 2):
        self.pooling_size = ps
        self.distance = 1
        self.gray_levels = 256
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]


    def read_img(self, fin:str):
        self.img = cv2.imread(fin, cv2.IMREAD_GRAYSCALE)


    def pooling(self):
        kernel = np.ones((self.pooling_size, self.pooling_size), np.float32) / (self.pooling_size**2)
        self.pooled_img = cv2.filter2D(self.img, -1, kernel)


    def imshow(self):
        fig, [ax0, ax1] = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(self.img)
        ax1.imshow(self.pooled_img)
        plt.show()


    def GLCM(self):

        self.glcm = cv2.calcGLCM(self.pooled_img.astype(np.int8),
                                 [distance],
                                 angles,
                                 gray_levels,
                                 symmetric=True,
                                 normed=True)
        entropy = cv2.GlcmProperties(glcm).entropy[0]
        print("GLCM entropy", entropy)


