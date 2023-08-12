#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk


# load sample image
image = img_as_ubyte(data.coins())

# disk(5) represents neighboring value of the circle with radius 5.
gradient_image = rank.gradient(image, disk(5))

# show the image
fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Original Image")

ax[1].imshow(gradient_image, cmap=plt.cm.gray)
ax[1].set_title("Gradient Image")

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
