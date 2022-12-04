#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import io
import imgproc


class texture():

    def __init__(self):
        pass

    def get_img(self):
        IO = io.io()
        IO.load_img()

        proc = imgproc.imgproc()
        proc.binarize()
