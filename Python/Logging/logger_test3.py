#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

class LoggerTest:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        #ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def make_log(self):
        self.logger.info("make_log method is executed.")
        self.logger.debug("make_log debug message.")


if __name__ == '__main__':
    lt = LoggerTest()
    lt.make_log()
