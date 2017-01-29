from __future__ import print_function
from math import sqrt
from astropy.io import fits
import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter
matplotlib.use('Qt5Agg')
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.restoration import denoise_wavelet

import matplotlib.pyplot as plt


def load_fits(fin):
    with fits.open(fin) as _f:
        scidata = np.array(_f[0].data, dtype=np.float64)
    return scidata


# image = data.hubble_deep_field()[0:500, 0:500]
# image_gray = rgb2gray(image)
# print(np.min(np.min(image_gray)), np.max(np.max(image_gray)))

image_gray = load_fits('4_3200_Phaethon_VIC_lp600_o_20161022_042047.042560_summed.fits')
image_gray /= np.max(np.max(image_gray))
# take log:
image_gray = np.log(image_gray)
# smooth:
image_gray = gaussian_filter(image_gray, sigma=2)
# denoise instead:
# image_gray = denoise_wavelet(image_gray, sigma=1e-8)

print('doing LoG')
blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

print('doing DoG')
blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

print('doing DoH')
blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

# # TODO: remove next line
# blobs_log, blobs_dog = blobs_doh, blobs_doh

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    # ax[idx].imshow(image, interpolation='nearest')
    ax[idx].imshow(image_gray, interpolation='nearest', origin='lower', cmap=plt.get_cmap('magma'))
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()