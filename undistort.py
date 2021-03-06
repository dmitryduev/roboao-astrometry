from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import argparse

from astropy.io import fits
from scipy.interpolate import griddata


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header:
        header = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header[_entry[0]] = _entry[1:]

    return header


def load_fits(fin, return_header=False):
    with fits.open(fin) as _f:
        _scidata = _f[0].data
    _header = get_fits_header(fin) if return_header else None

    return _scidata, _header


def export_fits(path, _data, _header=None):
    """
        Save fits file overwriting if exists
    :param path:
    :param _data:
    :param _header:
    :return:
    """
    if _header is not None:
        hdu = fits.PrimaryHDU(_data, header=_header)
    else:
        hdu = fits.PrimaryHDU(_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)


def map_xy_all_sky(p, M, sx, sy, xy_linspace):

    a_01, a_02, a_11, a_10, a_20, b_01, b_02, b_11, b_10, b_20 = p

    xv, yv = np.meshgrid(xy_linspace, xy_linspace, sparse=False, indexing='xy')
    vu, vv = np.zeros_like(xv), np.zeros_like(yv)

    for i in range(len(xy_linspace)):
        for j in range(len(xy_linspace)):
            x = xv[j, i]
            y = yv[j, i]
            f = a_01 * y + a_02 * y ** 2 + a_11 * x * y + a_10 * x + a_20 * x ** 2
            g = b_01 * y + b_02 * y ** 2 + b_11 * x * y + b_10 * x + b_20 * x ** 2
            uv = M * np.array([[(x + f) / sx],
                               [(y + g) / sy]])
            vu[j, i] = uv[0]
            vv[j, i] = uv[1]
            # print(i,j)

    return vu, vv


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='undistort frame')

    parser.add_argument('fits_in', metavar='fits_in',
                        action='store', help='path to image fits file.', type=str)

    parser.add_argument('-d', action='store_true', dest='drizzled', help='input image x2 oversampled by Drizzle?')

    args = parser.parse_args()

    # fits in:
    img, header = load_fits(args.fits_in, return_header=True)

    # mapping parameters:
    # 20170425:
    # mapping = np.array([2.0329677243543589e-08, 7.2072010293158654e-08,
    #                     -5.8616871850289164e-07, 2.0611255096058385e-04,
    #                     -1.1786163956914184e-05, 2.5133219448017527e-03,
    #                     -3.6051783118192822e-05, 7.2491660939103119e-06,
    #                     2.2510260984021737e-05, -2.8895716369256968e-05])
    # 20170426:
    mapping = np.array([-1.2024145389640598e-04, 1.0034774658106170e-06,  -2.9297259719493117e-06,
                        1.7502587328546042e-04,   1.2884723893270480e-06,   6.1281784372623858e-03,
                        -5.3855711448417457e-05,   1.7855794831797325e-06,  -3.0533154887920266e-04,
                        -5.4149126150957471e-05])

    '''
    # 20170427
    # mapping, bootstrapped on 10 noise realisations
    array([ -1.6634336860881237e-04,   1.3600851843242876e-06,
        -3.5528069763799320e-06,   2.1959694282964819e-04,
         1.9730137649319920e-06,   6.1190346731966033e-03,
        -5.3582387047539367e-05,   2.2548006932596036e-06,
        -4.1552594573179098e-04,  -5.4396656934798761e-05]),
    # bootstrapped error
    array([  1.3165895694674479e-04,   1.1884787074201178e-06,
         1.2906920441767467e-06,   7.5694718227687730e-05,
         1.6976827325106898e-06,   2.6358432506499519e-04,
         1.3920528073223987e-06,   1.6712043180243759e-06,
         2.0956358055545526e-04,   1.1401770365211843e-06])
    '''

    # linear transformation matrix:
    M = np.matrix([[-9.9411769544196640e-06, 9.3382713752932988e-08],
                   [1.6755094972110852e-08, 9.6818838309733057e-06]])

    # scale matrix:
    Q, R = np.linalg.qr(M)

    nx = img.shape[0]
    pixel_range = np.linspace(-nx // 2 + 1, nx // 2, nx)
    if not args.drizzled:
        vu, vv = map_xy_all_sky(mapping, M, -R[0, 0], R[1, 1], pixel_range)
    else:
        vu, vv = map_xy_all_sky(mapping, M / 2, -R[0, 0] / 2, R[1, 1] / 2, pixel_range)

    # interpolate into a regular grid
    # xx, yy = np.mgrid[-nx // 2 + 1: nx // 2: 1, -nx // 2 + 1: nx // 2: 1]
    xx, yy = np.mgrid[0:nx:1, 0:nx:1] - nx // 2

    img_no_distortion = griddata((np.ravel(vu), np.ravel(vv)), np.ravel(img),
                                 (xx, yy), method='cubic', fill_value=0)
    # print(img_no_distortion.shape)

    # export undistorted image
    export_fits(args.fits_in.replace('.fits', '.undistorted.fits'), np.rot90(np.flipud(img_no_distortion), 3))
