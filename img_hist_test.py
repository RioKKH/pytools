#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import netpbmfile
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.filters import gaussian
from skimage.filters import threshold_isodata
from skimage.filters import threshold_minimum
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li


filters = {
    "isodata":threshold_isodata,
    "minimum":threshold_minimum,
    "mean": threshold_mean,
    "otsu": threshold_otsu,
    "yen": threshold_yen,
    "li": threshold_li
}


class xo2():

    def __init__(self, fin:str=None) -> None:
        if fin is not None:
            self.load_img(fin)
        else:
            pass

    def load_img(self, fin:str) -> None:
        p = Path()
        self.fname = fin
        self.fin = p / fin

        if self.fin.suffix == ".pgm":
            self.img = netpbmfile.imread(fin)
        elif self.fin.suffix == ".jpg" or ".png":
            self.img = io.imread(fin)
        else:
            print("Suffix of the file must be PGM, JPG, or PNG")
            sys.exit(1)


    def invert(self) -> None:
        self.img = np.invert(self.img)


    def apply_gaussian_filter(self,
                              sigma=1.0,
                              preserve_range=True) -> None:

        self.gimg = gaussian(self.img,
                             sigma,
                             preserve_range=preserve_range)


    def binarize(self, kind:str="otsu",
                 white_bg:bool=False, remove:bool=False) -> None:
    
        if hasattr(self, "gimg"):
            temp_bimg = self.gimg > filters[kind](self.gimg)
            if white_bg:
                self.bimg = np.invert(temp_bimg)
            else:
                self.bimg = temp_bimg
            if remove:
                    self.bimg = clear_border(self.bimg)
        else:
            print("No gimg found")


    def get_property(self):
        self.labeld_img, self.n = label(self.bimg, return_num = True)
        self.image_label_overlay = label2rgb(self.labeld_img,
                                             image=self.img,
                                             bg_label=0)
        self.regions = regionprops(self.labeld_img, intensity_image = self.gimg)
        self.ecce_list = np.array([self.regions[i].eccentricity
                                   for i in range(0, len(self.regions))])


    def find_center_beamlet(self):
        d = {i:self.regions[i].area for i in range(self.n)}
        largest = sorted(d.items(), key = lambda x: x[1])[0]
        self.cx, self.cy = self.regions[largest[0]].centroid
        print(self.cx, self.cy)


    def show(self, kind='r', origin='upper') -> None:
        if kind == 'r':
            plt.imshow(self.img, origin=origin, extent=[0, self.img.shape[0],
                                                        0, self.img.shape[1]])
        elif kind == 'g':
            plt.imshow(self.gimg, origin=origin, extent=[0, self.gimg.shape[0],
                                                         0, self.gimg.shape[1]])
        elif kind == 'b':
            plt.imshow(self.bimg, origin=origin, extent=[0, self.bimg.shape[0],
                                                         0, self.bimg.shape[1]])
        elif kind == 'l':
            plt.imshow(self.image_label_overlay,
                       origin=origin, 
                       extent=[0, self.bimg.shape[0], 0, self.bimg.shape[1]])

        plt.show()


def test_threshold_method(imgarray, depth=256):
    isodata_threshold = threshold_isodata(imgarray)
    minimum_threshold = threshold_minimum(imgarray)
    mean_threshold    = threshold_mean(imgarray)
    otsu_threshold    = threshold_otsu(imgarray)
    yen_threshold     = threshold_yen(imgarray)
    li_threshold      = threshold_li(imgarray)

    plt.hist(imgarray.ravel(), range(depth))
    plt.axvline(x=isodata_threshold, label='Isodata', c='blue', alpha=0.5)
    plt.axvline(x=minimum_threshold, label='Minimum', c='red', alpha=0.5)
    plt.axvline(x=mean_threshold, label='Mean', c='green', alpha=0.5)
    plt.axvline(x=otsu_threshold, label='Otsu-san', c='lime', alpha=0.5)
    plt.axvline(x=yen_threshold, label='Yen', c='pink', alpha=0.5)
    plt.axvline(x=li_threshold, label='Li', c='orange', alpha=0.5)

    plt.legend()
    plt.show()
    

