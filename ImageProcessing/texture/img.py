#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import netpbmfile
import matplotlib.pyplot as plt
import cv2

class img():

    def __init__(self):
        pass

    def imread(self, fin:str):
        p = Path()

        self.fname = fin
        self.fpath = p / fin
        self.ftype = self.fpath.suffix

        if self.ftype == ".pgm":
            self.img = netpbmfile.imread(self.fpath)

        elif self.ftype == ".jpg" or ".png":
            self.img = io.imread(self.fpath)

        else:
            print("File must be PGM, JPG, or PNG")
            sys.exit(1)

    def imwrite(self, fout:str = "out"):
        pass

    def get_img(self):
        return self.img

    def imshow(self, which="dst"):
        if which == "org":
            cv2.imshow("Original Image", self.img)

        elif which == "dst":
            cv2.imshow("Processed Image", self.dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



