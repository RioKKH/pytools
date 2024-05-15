#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(
    level=logging.DEBUG,
    #level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    filename="my_app.log",
)

logger = logging.getLogger(__name__)

logger.debug("DEBUG message")
logger.info("INFO message")

def my_function():
    logger.info("Conducted my_function")

my_function()

