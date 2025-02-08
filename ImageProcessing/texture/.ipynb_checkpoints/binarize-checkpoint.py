#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
こちらのサイトのコードを使わせていただいた
https://pystyle.info/opencv-image-binarization/
"""


import cv2
import numpy as numpy
from IPython.display import Image, display
from ipywidgets import widgets
from matplotlib import pyplot as plt


def imshow(img):
    """ Output the image on the Jupyter-labs
    """
    encoded = cv2.imencode(".bmp", img)[1]
    display(Image(encoded))


def process(thresh, type_, auto):
    """ Run binalization then output the result.
    """
    type_ = eval(type_)
    auto = eval(auto)
    if auto:
        type_ += auto
    ret, binary = cv2.threshold(img, thresh, maxval=255, type=type_)
    imshow(binary)


param_widgets = {}

# pulldown menu to set the parameter "type"
options = [
    "cv2.THRESH_BINARY",
    "cv2.THRESH_BINARY_INV",
    "cv2.THRESH_TRUNC",
    "cv2.THRESH_TOZERO",
    "cv2.THRESH_TOZERO_INV",
]

param_widgets["type"] = widgets.Dropdown(options=options, description="type: ")

# settings if to use automatical threshold method
options = [
    "None",
    "cv2.THRESH_OTSU",
    "cv2.THRESH_TRIANGLE",
]

param_widgets["auto"] = widgets.Dropdown(options=options, description="auto: ")

for x in param_widgets.values():
    x.layout.width = "400px"

# load image
img = cv2.imread("gray.bmp", cv2.IMREAD_GRAYSCALE)

# show the widgets
widgets.interacive(process, **param_widgets)




