from __future__ import print_function
import argparse
from astropy import wcs
from astropy.io import fits
from collections import OrderedDict
import os
import numpy as np


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


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Add WCS data to the FITS header')

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

    x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20 = mapping

    nx, ny = img.shape

    w = wcs.WCS(naxis=2)
    w._naxis1 = nx
    w._naxis2 = ny
    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2

    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    # position of the tangential point on the detector [pix]
    # w.wcs.crpix = np.array([w.naxis1 // 2, w.naxis2 // 2])
    if args.drizzled:
        w.wcs.crpix = np.array([x_tan * 2.0, y_tan * 2.0])
    else:
        w.wcs.crpix = np.array([x_tan, y_tan])

    # sky coordinates of the tangential point; this is not known a priori. could set to TELRA/TELDEC?
    if ('TELRA' in header) and ('TELDEC' in header):
        w.wcs.crval = [header['TELRA'][0], header['TELDEC'][0]]
    else:
        w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    # linear mapping detector :-> focal plane [deg/pix]
    w.wcs.cd = np.array([[M_11, M_12],
                         [M_21, M_22]])
    if args.drizzled:
        w.wcs.cd /= 2

    # set up quadratic distortions [xy->uv and uv->xy]
    m = 2
    a = np.zeros((m + 1, m + 1), np.double)
    a[0, 2] = a_02
    a[1, 1] = a_11
    a[2, 0] = a_20
    b = np.zeros((m + 1, m + 1), np.double)
    b[0, 2] = b_02
    b[1, 1] = b_11
    b[2, 0] = b_20
    if args.drizzled:
        a /= 2
        b /= 2
    ap = np.zeros((m + 1, m + 1), np.double)
    bp = np.zeros((m + 1, m + 1), np.double)
    w.sip = wcs.Sip(a, b, ap, bp, w.wcs.crpix)

    # header['BLALA'] = 'LALA'
    # turn WCS object into header
    new_header = w.to_header(relax=True)
    # merge with old header:
    for key in header.keys():
        new_header[key] = header[key]

    export_fits(args.fits_in.replace('.fits', '.wcs.fits'), img, _header=new_header)
