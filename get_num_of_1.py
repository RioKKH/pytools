#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def load(fin:str):
    with open(fin, "r") as fr:
        for line in fr:
            check_num_of_one(line)


def check_num_of_one(line:str):
    print(line, len(line.strip()), line.count('0'), line.count('1'))



