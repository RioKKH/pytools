#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def mywatershed(image, markers):
    # Initialize the output image with the markers
    print(markers)
    output = np.copy(markers)

    # Get the indices of the pixels sorted by their intensity in the image
    indices = np.argsort(image, axis=None)

    # Initialize the union-find data structure
    parent = np.arange(image.size)
    rank = np.zeros_like(parent)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            elif rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[y_root] = x_root
                rank[x_root] += 1

    # Process each pixel in the order of their intensity
    for index in indices:
        x, y= np.unravel_index(index, image.shape)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx >= 0 and nx < image.shape[0] and ny >= 0 and ny < image.shape[1]:
                if output[nx, ny] > 0:
                    union(index, np.ravel_multi_index((nx, ny), image.shape))
        root = find(index)
        output[np.unravel_index(root, image.shape)] = output[x, y]

    return output

# Load the coins image from skimage
image = img_as_ubyte(data.coins())

# Compute the gradient of the image
gradient_image = rank.gradient(image, disk(5))

# Display the original image and the gradient image
fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(gradient_image, cmap=plt.cm.gray)
ax[1].set_title('Gradient Image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Compute the local maxima of the gradient image
coordinates = peak_local_max(gradient_image, min_distance=20, labels=image)

# Create an array of markers with the same size as the gradient image,
# filled with zeros
markers = np.zeros_like(gradient_image, dtype=int)

# Set the markers at the coordinates of the local maxima to their correspoding label
markers[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates)) + 1

# Apply the watershed algorithm to the gradient image using the markers
labels = watershed(gradient_image, markers)

# Display the original image and the segmentation result
fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Segmentation Result')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()



# Apply the my watershed algorighm to the gradient image using the markers
labels = mywatershed(gradient_image, markers)

# Display the original image and the gradient image
fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Segmentation Result')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
