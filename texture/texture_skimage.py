#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import os
#import sys
#from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


class texture():

    def __init__(self, ps = 2):
        self.pooling_size = ps

    def read_img(self, fin:str):
        self.img = io.imread(fin)
        self.pooled_img = np.zeros((self.img.shape[0] // self.pooling_size,
                                    self.img.shape[1] // self.pooling_size))

    def imshow(self):
        plt.imshow(self.img)
        plt.show()


    def pooling(self):
        for i in range(0, self.img.shape[0], self.pooling_size):
            for j in range(0, self.img.shape[1], self.pooling_size):
                self.pooled_img[i//self.pooling_size,
                                j//self.pooling_size] =\
                    np.mean(self.img[i:i+self.pooling_size,
                                     j:j+self.pooling_size])


    def GLCM(self):
        gray_levels = 256
        distance = 1
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        self.glcm = graycomatrix(self.pooled_img.astype(np.uint8),
                                 distances=[distance],
                                 angles=angles,
                                 levels=gray_levels,
                                 symmetric=True,
                                 normed=True)
        homogeneity = graycoprops(self.glcm, 'homogeneity')
        entropy = shannon_entropy(self.img)
        pentropy = shannon_entropy(self.pooled_img)
        print("GLCM Homogeneity", homogeneity)
        print("GLCM Shannon entropy", entropy, pentropy)


