#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import traceback
from pathlib import Path

import numpy as np
import netpbmfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import ndimage as ndi
from skimage import io
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.color import label2rgb, rgb2gray
from skimage.feature import peak_local_max
from skimage.feature import canny
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.filters import rank
from skimage.filters import gaussian
from skimage.filters import threshold_isodata
from skimage.filters import threshold_minimum
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from skimage.filters import try_all_threshold


filters = {
    "isodata":threshold_isodata,
    "minimum":threshold_minimum,
    "mean": threshold_mean,
    "otsu": threshold_otsu,
    "yen": threshold_yen,
    "li": threshold_li
}


class xo():

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
        self.cy, self.cx = self.regions[largest[0]].centroid
        print(self.cx, (511 - self.cy))
        return (self.cx, (511 - self.cy))


    def dump_vhdata(self, prefix='on_'):
        try:
            np.savetxt(prefix + 'hor.dat', self.img[int(self.cy), :])
            np.savetxt(prefix + 'ver.dat', self.img[:, int(self.cx)])
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def make_markers(self, mradius=5, gradius=2, binthreshold=30):
        self.markers = rank.gradient(self.gimg/self.gimg.max(), footprint=disk(mradius)) < binthreshold
        self.markers = ndi.label(self.markers)[0]
        print("Num of markers assigned: ", self.markers.max())

        self.gradient = rank.gradient(self.gimg/self.gimg.max(), footprint=disk(gradius))
        self.labels = watershed(self.gradient, self.markers)
        regions = regionprops(self.labels)
        print("Num of labels assigned: ", self.markers.max())

        fig = plt.figure(figsize=(8, 3))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.25, share_all=True)

        grid[0].imshow(self.gimg, cmap=plt.cm.gray)
        grid[1].imshow(self.gradient, cmap=plt.cm.nipy_spectral)
        grid[2].imshow(self.markers, cmap=plt.cm.nipy_spectral)
        #grid[3].imshow(self.img, cmap=plt.cm.gray)
        grid[3].imshow(self.labels)
        #grid[3].imshow(self.labels, cmap=plt.cm.prism, alpha=0.5)

        #grid[3].scatter(regions[0].centroid[1], regions[0].centroid[0], marker='o', s=2)
        grid[3].scatter(regions[1].centroid[1], regions[1].centroid[0], marker='o', s=2)
        grid[3].scatter(regions[2].centroid[1], regions[2].centroid[0], marker='o', s=2)
        #plt.scatter(self.cx, (511- self.cy), marker='o', s=2, c='red', alpha=0.7)

        t_list = ["Original", "Local Gradient", "Markers", "Segmented"]
        for i in range(len(t_list)):
            grid[i].axis("off")
            grid[i].set_title(t_list[i])

        plt.tight_layout()
        plt.show()

        np.savetxt("segmented.txt", self.labels)
        #plt.imshow(self.labels)
        #plt.savefig('segmented.png', dpi=150)

        return regions

    def hough_transformation(self, sigma=2, num=1, origin='upper'):
        self.img2 = (self.labels == 2)
        self.img3 = (self.labels == 3)
        self.gimg2 = gaussian(self.img2, sigma)
        self.gimg3 = gaussian(self.img3, sigma)
        self.edge2 = canny(self.gimg2)
        self.edge3 = canny(self.gimg3)
        hough_radii = np.arange(50, 150, 2)
        hough_res2 = hough_circle(self.edge2, hough_radii)
        hough_res3 = hough_circle(self.edge3, hough_radii)
        #print(hough_res2.shape, hough_res3.shape)
        accums2, cy2, cx2, radii2 = hough_circle_peaks(hough_res2, hough_radii, total_num_peaks=num)
        accums3, cy3, cx3, radii3 = hough_circle_peaks(hough_res3, hough_radii, total_num_peaks=num)

        fig, (ax0, ax1) = plt.subplots(ncols=2)

        ax0.imshow(self.edge2)

        for center_y, center_x, radius in zip(cx2, cy2, radii2):
            circy2, circx2 = circle_perimeter(center_y, center_x, radius)
            ax0.scatter(center_x, center_y, marker='o', s=2, color='red', alpha=0.8)
            ax0.plot(circx2, circy2, alpha=0.3, color='y')
        ax0.set_title("center_x: %d\ncenter_y: %d" %(center_x, center_y))

        ax1.imshow(self.edge3)

        for center_y, center_x, radius in zip(cx3, cy3, radii3):
            circy3, circx3 = circle_perimeter(center_y, center_x, radius)
            ax1.scatter(center_x, center_y, marker='o', s=2, color='red', alpha=0.8)
            ax1.plot(circx3, circy3, alpha=0.3, color='y')
        ax1.set_title("center_x: %d\ncenter_y: %d" %(center_x, center_y))

        plt.show()




    def run_watershed(self):
        distance = ndi.distance_transform_edt(self.bimg)
        coords = peak_local_max(distance, footprint=np.ones((12, 12)), labels=self.bimg)
        print("sum of local maxes: ", coords.sum())
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=self.bimg)

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,
                                            figsize=(9, 3),
                                            sharex=True, sharey=True)
        ax0.imshow(self.bimg, cmap=plt.cm.gray)
        ax0.set_title("Overlapping objects")
        ax1.imshow(-distance, cmap=plt.cm.gray)
        ax1.set_title("Distances")
        ax2.imshow(labels, cmap=plt.cm.nipy_spectral)
        ax2.set_title("Separated objects")

        ax0.set_axis_off()
        ax1.set_axis_off()
        ax2.set_axis_off()

        fig.tight_layout()
        plt.show()


    def show(self, kind='r', origin='upper') -> None:
        if kind == 'r':
            plt.imshow(self.img, origin=origin, extent=[0, self.img.shape[0],
                                                        0, self.img.shape[1]])
            plt.scatter(self.cx, (511- self.cy), marker='o', s=2, c='red', alpha=0.7)

        elif kind == 'g':
            plt.imshow(self.gimg, origin=origin, extent=[0, self.gimg.shape[0],
                                                         0, self.gimg.shape[1]])
        elif kind == 'b':
            plt.imshow(self.bimg, origin=origin, extent=[0, self.bimg.shape[0],
                                                         0, self.bimg.shape[1]])
            plt.scatter(self.cx, (511 - self.cy), marker='o', s=2, c='red', alpha=0.7)

        elif kind == 'l':
            plt.imshow(self.image_label_overlay,
                       origin=origin, 
                       extent=[0, self.bimg.shape[0], 0, self.bimg.shape[1]])

        plt.show()

    def plot3d(self):
        x = np.linspace(0, self.img.shape[0]-1, self.img.shape[0])
        y = np.linspace(0, self.img.shape[1]-1, self.img.shape[1])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(np.ravel(X), np.ravel(Y), np.ravel(self.img), s=10, marker='.')
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
    

def test_all(img:np.ndarray) -> None:
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()
