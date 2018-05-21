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


def map_xy_all_sky(p, xy_linspace, drizzled=False):

    x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20 = p

    M = np.matrix([[M_11, M_12], [M_21, M_22]])
    if drizzled:
        x_tan *= 2.0
        y_tan *= 2.0
        M /= 2.0
        a_02 /= 2
        a_11 /= 2
        a_20 /= 2
        b_02 /= 2
        b_11 /= 2
        b_20 /= 2

    Q, R = np.linalg.qr(M)

    sx, sy = -R[0, 0], R[1, 1]

    xv, yv = np.meshgrid(xy_linspace, xy_linspace, sparse=False, indexing='xy')
    vu, vv = np.zeros_like(xv), np.zeros_like(yv)

    for i in range(len(xy_linspace)):
        for j in range(len(xy_linspace)):
            delta_x = xv[j, i] - x_tan
            delta_y = yv[j, i] - y_tan
            uv = np.array(
                M * np.array([[delta_x + a_02 * delta_y ** 2 + a_11 * delta_x * delta_y + a_20 * delta_x ** 2],
                              [delta_y + b_02 * delta_y ** 2 + b_11 * delta_x * delta_y + b_20 * delta_x ** 2]]))
            vu[j, i] = uv[0] / sx
            vv[j, i] = uv[1] / sy
            # print(i,j)

    return vu, vv


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='transform frame from detector space to tangent plane.\n' +
                                                 '(x,y)[pix] -> (u,v)[deg]\n' +
                                                 'First apply quadratic distortion map to (x,y),\n' +
                                                 'Then linear mapping M accounting for reflection, skew, and rotation.')

    parser.add_argument('fits_in', metavar='fits_in',
                        action='store', help='path to image fits file.', type=str)

    parser.add_argument('-d', action='store_true', dest='drizzled', help='input image x2 oversampled by Drizzle?')

    args = parser.parse_args()

    # fits in:
    img, header = load_fits(args.fits_in, return_header=True)

    # mapping parameters:
    # 20170620:
    # x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20
    mapping = np.array([-1.7097340244173049e+00, -2.9573349942689418e+00,
                        -9.9243124794318175e-06, 9.1942882975760912e-08,
                        -1.9338413987785716e-08, 9.7191812668049799e-06,
                        -7.3628182254640821e-07, -2.9041549472251136e-07, 2.6488080053108772e-07,
                        -5.5062280886481794e-05, 2.7321332070856352e-07, -5.3791223995525186e-05])

    nx = img.shape[0]
    pixel_range = np.linspace(-nx // 2 + 1, nx // 2, nx)
    vu, vv = map_xy_all_sky(mapping, pixel_range, args.drizzled)

    # interpolate into a regular grid
    # xx, yy = np.mgrid[-nx // 2 + 1: nx // 2: 1, -nx // 2 + 1: nx // 2: 1]
    xx, yy = np.mgrid[0:nx:1, 0:nx:1] - nx // 2

    img_no_distortion = griddata((np.ravel(vu), np.ravel(vv)), np.ravel(img),
                                 (xx, yy), method='cubic', fill_value=0)
    # print(img_no_distortion.shape)

    # export undistorted image
    export_fits(args.fits_in.replace('.fits', '.undistorted.fits'), np.rot90(np.flipud(img_no_distortion), 3))
    # export_fits(args.fits_in.replace('.fits', '.undistorted.fits'), img_no_distortion)
