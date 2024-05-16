#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("my_app.log")
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"
)

# Set the format to the handler
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# log output
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")

# logging called in a function
def my_function():
    logger.info("logger defined in the function is conducted.")

my_function()

