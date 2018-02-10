from __future__ import print_function
import sewpy
import numpy as np
import os
import inspect
from skimage import exposure, img_as_float
from copy import deepcopy
from scipy.optimize import leastsq
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
from collections import OrderedDict
from astroquery.query import suspend_cache
from astroquery.vizier import Vizier
from itertools import combinations
from numba import jit
from time import time as _time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
# # sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
# plt.close('all')
# sns.set_context('talk')
import aplpy
import image_registration
np.set_printoptions(16)


# initialize Vizier object
with suspend_cache(Vizier):
    viz = Vizier()
    viz.ROW_LIMIT = -1
    viz.TIMEOUT = 30


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


def scale_image(image, correction='local'):
    """

    :param image:
    :param correction: 'local', 'log', or 'global'
    :return:
    """
    # scale image for beautification:
    scidata = deepcopy(image)
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata / norm * 65535)

    # add more contrast to the image:
    if correction == 'log':
        return exposure.adjust_log(img_as_float(scidata/norm) + 1, 1)
    elif correction == 'global':
        p_1, p_2 = np.percentile(scidata, (5, 100))
        # p_1, p_2 = np.percentile(scidata, (1, 20))
        return exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
    elif correction == 'local':
        # perform local histogram equalization instead:
        return exposure.equalize_adapthist(scidata, clip_limit=0.03)
    else:
        raise Exception('Contrast correction option not recognized')


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def generate_image(xy, mag, xy_ast=None, mag_ast=None, exp=None, nx=2048, ny=2048, psf=None):
    """

    :param xy:
    :param mag:
    :param xy_ast:
    :param mag_ast:
    :param exp: exposure in seconds to 'normalize' streak
    :param nx:
    :param ny:
    :param psf:
    :return:
    """

    if isinstance(xy, list):
        xy = np.array(xy)
    if isinstance(mag, list):
        mag = np.array(mag)

    image = np.zeros((ny, nx))

    # let us assume that a 6 mag star would have a flux of 10^9 counts
    flux_0 = 1e9
    # scale other stars wrt that:
    flux = flux_0 * 10 ** (0.4 * (6 - mag))
    # print(flux)

    # add stars to image
    for k, (i, j) in enumerate(xy):
        if i < nx and j < ny:
            image[int(j), int(i)] = flux[k]

    if exp is None:
        exp = 1.0
    # add asteroid
    if xy_ast is not None and mag_ast is not None:
        flux = flux_0 * 10 ** (0.4 * (6 - mag_ast))
        # print(flux)
        xy_ast = np.array(xy_ast, dtype=np.int)
        line_points = get_line(xy_ast[0, :], xy_ast[1, :])
        for (i, j) in line_points:
            if i < nx and j < ny:
                image[int(j), int(i)] = flux / exp

    if psf is None:
        # Convolve with a gaussian
        image = gaussian_filter(image, 7 * nx/2e3)
    else:
        # convolve with a (model) psf
        # fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(nthreads=4, use_numpy_fft=False)
        # image = convolve_fft(image, psf, fftn=fftn, ifftn=ifftn)
        image = convolve_fft(image, psf)

    return image


def plot_field(target, window_size, _model_psf, grid_stars, num_pix=1024, _highlight_brighter_than_mag=None,
               _display_magnitude_labels=False, scale_bar=False, scale_bar_size=20,
               _display_plot=False, _save_plot=False, path='./', name='field'):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = int(num_pix * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    w._naxis2 = int(num_pix * (window_size[1] * 180.0 / np.pi * 3600) / 36)
    # w._naxis1 = int(1024 * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    # w._naxis2 = int(1024 * (window_size[1] * 180.0 / np.pi * 3600) / 36)
    # w._naxis1 = int(2048 * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    # w._naxis2 = int(2048 * (window_size[1] * 180.0 / np.pi * 3600) / 36)
    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2

    if w.naxis1 > 20000 or w.naxis2 > 20000:
        print('image too big to plot')
        return

    # Set up a tangential projection
    w.wcs.equinox = 2000.0
    # position of the tangential point on the detector [pix]
    w.wcs.crpix = np.array([w.naxis1 // 2, w.naxis2 // 2])
    # sky coordinates of the tangential point
    w.wcs.crval = [target.ra.deg, target.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # linear mapping detector :-> focal plane [deg/pix]
    # actual:
    # w.wcs.cd = np.array([[-4.9653758578816782e-06, 7.8012027500556068e-08],
    #                      [8.9799574245829621e-09, 4.8009647689165968e-06]])
    # with RA inverted to correspond to previews
    if num_pix == 2048:
        # for upsampled 2048x2048
        # w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
        #                      [8.9799574245829621e-09, 4.8009647689165968e-06]])
        # new:
        w.wcs.cd = np.array([[4.9938707892035111e-06,   2.7601870670659687e-08],
                             [7.7159226739276786e-09,   4.8389368351465302e-06]])
    else:
        # 1024x1024
        # w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
        #                      [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2
        # new:
        w.wcs.cd = np.array([[4.9938707892035111e-06, 2.7601870670659687e-08],
                             [7.7159226739276786e-09, 4.8389368351465302e-06]]) * 2
    # w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
    #                      [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2
    # w.wcs.cd = np.array([[5e-06, 1e-8],
    #                      [1e-8, 5e-06]]) * 2
    # print(w.wcs.cd)
    # [[-9.9628143319575235e-06,   6.0168523927372889e-08]
    #  [-1.1665005567301751e-07,   9.5769294337527134e-06]]

    # set up quadratic distortions [xy->uv and uv->xy]
    # w.sip = wcs.Sip(a=np.array([-1.7628536101583434e-06, 5.2721963537675933e-08, -1.2395119995283236e-06]),
    #                 b=np.array([2.5686775443756446e-05, -6.4405711579912514e-06, 3.6239787339845234e-05]),
    #                 ap=np.array([-7.8730574242135546e-05, 1.6739809945514789e-06,
    #                              -1.9638469711488499e-08, 5.6147572815095856e-06,
    #                              1.1562096854108367e-06]),
    #                 bp=np.array([1.6917947345178044e-03,
    #                              -2.6065393907218176e-05, 6.4954883952398105e-06,
    #                              -4.5911421583810606e-04, -3.5974854928856988e-05]),
    #                 crpix=w.wcs.crpix)

    ''' create a [fake] simulated image '''
    # apply linear transformation only:
    pix_stars = np.array(w.wcs_world2pix(grid_stars['RA_ICRS'], grid_stars['DE_ICRS'], 0)).T
    # apply linear + SIP:
    # pix_stars = np.array(w.all_world2pix(grid_stars['_RAJ2000'], grid_stars['_DEJ2000'], 0)).T
    # pix_stars = np.array(w.all_world2pix(grid_stars['RA_ICRS'], grid_stars['DE_ICRS'], 0)).T
    mag_stars = np.array(grid_stars['__Gmag_'])
    # print(pix_stars)
    # print(mag_stars)

    # tic = _time()
    sim_image = generate_image(xy=pix_stars, mag=mag_stars,
                               nx=w.naxis1, ny=w.naxis2, psf=_model_psf)
    # print(_time() - tic)

    # tic = _time()
    # convert simulated image to fits hdu:
    hdu = fits.PrimaryHDU(sim_image, header=w.to_header())

    ''' plot! '''
    # plt.close('all')
    # plot empty grid defined by wcs:
    # fig = aplpy.FITSFigure(w)
    # plot fake image:
    fig = aplpy.FITSFigure(hdu)

    # fig.set_theme('publication')

    fig.add_grid()

    fig.grid.show()
    fig.grid.set_color('gray')
    fig.grid.set_alpha(0.8)

    ''' display field '''
    # fig.show_colorscale(cmap='viridis')
    fig.show_colorscale(cmap='magma')
    # fig.show_grayscale()
    # fig.show_markers(grid_stars[cat]['_RAJ2000'], grid_stars[cat]['_DEJ2000'],
    #                  layer='marker_set_1', edgecolor='white',
    #                  facecolor='white', marker='o', s=30, alpha=0.7)

    # highlight stars bright enough to serve as tip-tilt guide stars:
    if _highlight_brighter_than_mag is not None:
        mask_bright = mag_stars <= float(_highlight_brighter_than_mag)
        if np.max(mask_bright) == 1:
            fig.show_markers(grid_stars[mask_bright]['RA_ICRS'], grid_stars[mask_bright]['DE_ICRS'],
                             layer='marker_set_2', edgecolor=plt.cm.Oranges(0.9),
                             facecolor=plt.cm.Oranges(0.8), marker='+', s=50, alpha=0.9, linewidths=1)

    # show labels with magnitudes
    if _display_magnitude_labels:
        for star in grid_stars:
            fig.add_label(star['RA_ICRS'], star['DE_ICRS'], '{:.1f}'.format(star['__Gmag_']),
                          color=plt.cm.Oranges(0.4), horizontalalignment='right')

    # add scale bar
    if scale_bar:
        fig.add_scalebar(length=scale_bar_size * u.arcsecond)
        fig.scalebar.set_alpha(0.7)
        fig.scalebar.set_color('white')
        fig.scalebar.set_label('{:d}\"'.format(scale_bar_size))

    # remove frame
    fig.frame.set_linewidth(0)
    # print(_time() - tic)

    if _display_plot:
        plt.show()

    if _save_plot:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.save(os.path.join(path, '{:s}.png'.format(name)))
        fig.close()

    return pix_stars, mag_stars


def make_image(target, window_size, _model_psf, pix_stars, mag_stars, num_pix=1024):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = int(num_pix * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    w._naxis2 = int(num_pix * (window_size[1] * 180.0 / np.pi * 3600) / 36)
    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2

    if w.naxis1 > 20000 or w.naxis2 > 20000:
        print('image too big to plot')
        return

    # Set up a tangential projection
    w.wcs.equinox = 2000.0
    # position of the tangential point on the detector [pix]
    w.wcs.crpix = np.array([w.naxis1 // 2, w.naxis2 // 2])
    # sky coordinates of the tangential point
    w.wcs.crval = [target.ra.deg, target.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # linear mapping detector :-> focal plane [deg/pix]
    # actual:
    # w.wcs.cd = np.array([[-4.9653758578816782e-06, 7.8012027500556068e-08],
    #                      [8.9799574245829621e-09, 4.8009647689165968e-06]])
    # with RA inverted to correspond to previews
    # for upsampled 2048x2048
    # w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
    # [8.9799574245829621e-09, 4.8009647689165968e-06]])
    # 1024x1024
    # w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
    #                      [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2
    # if num_pix == 2048:
    #     # for upsampled 2048x2048
    #     w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
    #                          [8.9799574245829621e-09, 4.8009647689165968e-06]])
    # else:
    #     # 1024x1024
    #     w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
    #                          [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2

    ''' create a [fake] simulated image '''
    # tic = _time()
    sim_image = generate_image(xy=pix_stars, mag=mag_stars, nx=w.naxis1, ny=w.naxis2, psf=_model_psf)

    return sim_image


def residual(p, x, y):
    """
        Simultaneously solve for RA_tan, DEC_tan, M, and 2nd-order distortion params
    :param p: RA_tan, Dec_tan, x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20
    :param x: star pixel positions on the detector
    :param y: star RA's and Dec's

    :return: estimated_map(x) - y
    """
    # tangent point RA/Dec:
    RA_tan = p[0]
    Dec_tan = p[1]

    # 3d position of the tangent point (on a unit sphere)
    t = np.array((np.cos(RA_tan * np.pi / 180.0) * np.cos(Dec_tan * np.pi / 180.0),
                  np.sin(RA_tan * np.pi / 180.0) * np.cos(Dec_tan * np.pi / 180.0),
                  np.sin(Dec_tan * np.pi / 180.0)))

    # position of the tangent point on the detector WRT the center of the latter:
    x_tan = p[2]
    y_tan = p[3]

    # linear relationship between native and pixel coordinates [deg/pix]
    M = np.matrix([[p[4], p[5]], [p[6], p[7]]])

    # 2nd-order distortion parameters:
    a_02, a_11, a_20, b_02, b_11, b_20 = p[8:]

    # 3d basis vectors
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)

    # apply the estimated map to the measured detector positions
    y_C = []

    for (x_i, y_i) in x:
        delta_x = x_i - x_tan
        delta_y = y_i - y_tan
        uv = np.array(M * np.array([[delta_x + a_02 * delta_y**2 + a_11 * delta_x * delta_y + a_20 * delta_x**2],
                                    [delta_y + b_02 * delta_y**2 + b_11 * delta_x * delta_y + b_20 * delta_x**2]]))

        # R_i = t + np.pi / 180.0 * np.array([uv[0][0], uv[1][0], 0])
        R_i = t + np.pi / 180.0 * (uv[0][0] * u + uv[1][0] * v)
        r_i = R_i / np.linalg.norm(R_i)

        RA_i = 180.0/np.pi * np.arctan2(np.dot(r_i, j), np.dot(r_i, i))
        if RA_i < 0:
            RA_i += 360.0
        Dec_i = 180.0/np.pi * np.arcsin(np.dot(r_i, k))

        y_C.append([RA_i, Dec_i])

    y_C = np.squeeze(np.array(y_C))

    # return np.linalg.norm(y.T - (M_m1 * UV.T + x_tan), axis=0)
    return np.linalg.norm(y.T - y_C.T, axis=0)


def fit_bootstrap(_residual, p0, datax, datay, yerr_systematic=0.0, n_samp=100, _scaling=None):

    # Fit first time
    _p = leastsq(_residual, p0, args=(datax, datay), full_output=True, ftol=1.49012e-13, xtol=1.49012e-13)

    pfit, perr = _p[0], _p[1]

    # Get the stdev of the residuals
    residuals = _residual(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # n_samp random data sets are generated and fitted
    ps = []
    # print('lala')
    # print(datay)
    for ii in range(n_samp):

        randomDelta = np.random.normal(0., sigma_err_total, size=datax.shape)
        # print(ii)
        randomdataX = datax + randomDelta
        # print(randomDelta)
        # raw_input()

        _p = leastsq(_residual, p0, args=(randomdataX, datay), full_output=True,
                     ftol=1.49012e-13, xtol=1.49012e-13, diag=_scaling)
        randomfit, randomcov = _p[0], _p[1]

        ps.append(randomfit)

    ps = np.array(ps)
    mean_pfit = np.mean(ps, 0)

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval
    Nsigma = 1.

    err_pfit = Nsigma * np.std(ps, 0)

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit

    return pfit_bootstrap, perr_bootstrap


if __name__ == '__main__':
    # path_in = '/Users/dmitryduev/_caltech/roboao/_faint_reductions/20170211/0_M13_VIC_Si_o_20170211_122715.043747/'
    # # fits_in = '100p.fits'
    # fits_in = '0_M13_VIC_Si_o_20170211_122715.043747_blind_decnv.fits'

    path_in = '/Users/dmitryduev/_caltech/roboao/_faint_reductions/20170604/0_M13_VIC_Si_o_20170604_084940.042100/'
    # fits_in = '0_M13_VIC_Si_o_20170604_084940.042100_blind_decnv.fits'
    fits_in = '0_M13_VIC_Si_o_20170604_084940.042100_blind_decnv_0.fits'

    # path_in = abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    # fits_in = '0_M13_VIC_Si_o_20170604_084940.042100_blind_decnv.fits'

    # see /usr/local/Cellar/sextractor/2.19.5/share/sextractor/default.sex

    # for drizzled:
    # sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "X2_IMAGE", "Y2_IMAGE", "XY_IMAGE",
    #                         "XWIN_IMAGE", "YWIN_IMAGE",
    #                         "FLUX_AUTO", "FLUXERR_AUTO",
    #                         "A_IMAGE", "B_IMAGE", "FWHM_IMAGE",
    #                         "FLAGS", "FLAGS_WEIGHT", "FLUX_RADIUS"],
    #                 config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '4.0'},
    #                 sexpath="sex")
    # for deconvolved:
    sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "X2_IMAGE", "Y2_IMAGE", "XY_IMAGE",
                            "XWIN_IMAGE", "YWIN_IMAGE",
                            "FLUX_AUTO", "FLUXERR_AUTO",
                            "A_IMAGE", "B_IMAGE", "FWHM_IMAGE",
                            "FLAGS", "FLAGS_WEIGHT", "FLUX_RADIUS"],
                    config={"BACK_SIZE": 32, "THRESH_TYPE": "ABSOLUTE", "CLEAN": "N",
                            "WEIGHT_GAIN": "N", "FILTER": "Y", "SATUR_LEVEL": 1e10,
                            "FILTER_NAME": "/usr/local/opt/sextractor/share/sextractor/gauss_3.0_7x7.conv",
                            "FILTER_THRESH": 2.0, "ANALYSIS_THRESH": 1.5, "SEEING_FWHM": 0.3,
                            "DETECT_MINAREA": 3, 'DETECT_THRESH': 40.0},
                    sexpath="sex")

    out = sew(os.path.join(path_in, fits_in))
    # # sort according to FWHM
    # out['table'].sort('FWHM_IMAGE')
    # sort according to raw flux
    out['table'].sort('FLUX_AUTO')
    # descending order: first is brightest
    out['table'].reverse()

    # get source pixel positions:
    pix_det = np.vstack((out['table']['X_IMAGE'], out['table']['Y_IMAGE'])).T
    pix_det_err = np.vstack((out['table']['X2_IMAGE'], out['table']['Y2_IMAGE'], out['table']['XY_IMAGE'])).T
    mag_det = np.array(out['table']['FLUX_AUTO'])

    # print detections:
    print(out['table'])  # This is an astropy table.
    # print('detected {:d} sources'.format(len(out['table'])))
    # print(np.mean(out['table']['A_IMAGE']), np.mean(out['table']['B_IMAGE']))
    # print(np.median(out['table']['A_IMAGE']), np.median(out['table']['B_IMAGE']))

    # load first image frame from the fits file
    preview_img, header = load_fits(os.path.join(path_in, fits_in), return_header=True)
    # print(preview_img.shape)
    # scale with local contrast optimization for preview:
    # preview_img = scale_image(preview_img, correction='local')
    # preview_img = scale_image(preview_img, correction='log')
    # preview_img = scale_image(preview_img, correction='global')

    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot detected objects:
    for i, _ in enumerate(out['table']['XWIN_IMAGE']):
        ax.plot(out['table']['X_IMAGE'][i]-1, out['table']['Y_IMAGE'][i]-1,
                'o', markersize=out['table']['FWHM_IMAGE'][i]/2,
                markeredgewidth=2.5, markerfacecolor='None', markeredgecolor=plt.cm.Greens(0.7),
                label='SExtracted')
        # ax.annotate(i, (out['table']['X_IMAGE'][i]+40, out['table']['Y_IMAGE'][i]+40),
        #             color=plt.cm.Blues(0.3), backgroundcolor='black')

    # ax.imshow(preview_img, cmap='gray', origin='lower', interpolation='nearest')
    # ax.imshow(preview_img, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
    ax.imshow(np.sqrt(np.sqrt(preview_img)), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    plt.grid('off')

    # save full figure
    # fname_full = '{:s}_full.png'.format(_obs)
    # if not (os.path.exists(_path_out)):
    #     os.makedirs(_path_out)
    # plt.savefig(os.path.join(_path_out, fname_full), dpi=300)

    # plt.show()

    ''' get stars from catalogue, create fake images, then cross correlate them '''
    # stars in the field (without mag cut-off):
    # print(header['OBJRA'][0], header['OBJDEC'][0])
    # star_sc = SkyCoord(ra=header['OBJRA'][0], dec=header['OBJDEC'][0], unit=(u.hourangle, u.deg), frame='icrs')
    star_sc = SkyCoord(ra=header['TELRA'][0], dec=header['TELDEC'][0], unit=(u.hourangle, u.deg), frame='icrs')
    print('nominal FoV center:', star_sc)

    # solved for:
    star_sc = SkyCoord(ra=2.5041593116369444e+02, dec=3.6456177994034121e+01, unit=(u.deg, u.deg), frame='icrs')
    # star_sc = SkyCoord(ra=2.5041567159848793e+02, dec=3.6455898328568495e+01, unit=(u.deg, u.deg), frame='icrs')

    # search radius: " -> rad
    fov_size_ref = 100 * np.pi / 180.0 / 3600

    catalog = 'I/337/gaia'
    stars_table = viz.query_region(star_sc, width=fov_size_ref * u.rad,
                                   height=fov_size_ref * u.rad,
                                   catalog=catalog, cache=False)
    fov_stars = stars_table[catalog]
    print(fov_stars)

    pix_ref, mag_ref = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                  grid_stars=fov_stars, num_pix=preview_img.shape[0], _highlight_brighter_than_mag=None,
                                  scale_bar=False, scale_bar_size=20, _display_plot=False, _save_plot=False,
                                  path='./', name='field')
    if False:
        print(pix_ref)

    # brightest 20:
    # tic = _time()
    # quads_reference = triangulate(xy_grid, cut=30, max_pix=1500)
    # print(_time() - tic)

    # print(len(quads_detected), len(quads_reference))

    ''' detect shift '''
    fov_size_det = 36 * np.pi / 180.0 / 3600
    mag_det /= np.max(mag_det)
    mag_det = -2.5*np.log10(mag_det)
    # add (pretty arbitrary) baseline
    mag_det += np.min(mag_ref)
    # print(mag_det)

    # detected = make_image(target=star_sc, window_size=[fov_size_det, fov_size_det], _model_psf=None,
    #                       pix_stars=pix_det, mag_stars=mag_det)
    naxis_det = int(preview_img.shape[0] * (fov_size_det * 180.0 / np.pi * 3600) / 36)
    naxis_ref = int(preview_img.shape[0] * (fov_size_ref * 180.0 / np.pi * 3600) / 36)
    if False:
        print(naxis_det, naxis_ref)
    # effectively shift detected positions to center of ref frame to reduce distortion effect
    pix_det_ref = pix_det + np.array([naxis_ref//2, naxis_ref//2]) - np.array([naxis_det//2, naxis_det//2])
    detected = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                          pix_stars=pix_det_ref, mag_stars=mag_det, num_pix=preview_img.shape[0])
    reference = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                           pix_stars=pix_ref, mag_stars=mag_ref, num_pix=preview_img.shape[0])

    # register shift: pixel precision first
    from skimage.feature import register_translation
    shift, error, diffphase = register_translation(reference, detected, upsample_factor=1)
    print('pixel precision offset:', shift, error)
    # shift, error, diffphase = register_translation(reference, detected, upsample_factor=2)
    # print('subpixel precision offset:', shift, error)

    # match!
    matched = []
    mask_matched = []
    for si, s in enumerate(pix_det_ref):
        s_shifted = s + np.array(shift[::-1])

        pix_distance = np.min(np.linalg.norm(pix_ref - s_shifted, axis=1))
        if False:
            print(pix_distance)

        # note: because of larger distortion in the y-direction, pix diff there is larger than in x
        # if pix_distance < 25 * preview_img.shape[0] / 1024:  # 25:
        if pix_distance < 20:
            min_ind = np.argmin(np.linalg.norm(pix_ref - s_shifted, axis=1))

            # note: errors in Gaia position are given in mas, so convert to deg by  / 1e3 / 3600
            matched.append(np.hstack([pix_det[si], pix_det_err[si],
                                        np.array([fov_stars['RA_ICRS'][min_ind],
                                                  fov_stars['DE_ICRS'][min_ind],
                                                  fov_stars['e_RA_ICRS'][min_ind] / 1e3 / 3600,
                                                  fov_stars['e_DE_ICRS'][min_ind] / 1e3 / 3600,
                                                  0.0])]))
                                                  # fov_stars['RADEcor'][min_ind]])]))
            mask_matched.append(min_ind)

    matched = np.array(matched)
    if False:
        print('matched objects:')
        print(matched)
    print('total matched:', len(matched))
    median_e_RA_ICRS = np.median(matched[:, 7]) * 1e3 * 3600  # mas
    median_e_DE_ICRS = np.median(matched[:, 8]) * 1e3 * 3600  # mas
    print('median astrometric errors in RA, Dec, and sqrt(RA^2 + Dec^2) in mas:',
          median_e_RA_ICRS, median_e_DE_ICRS, np.sqrt(median_e_RA_ICRS**2 + median_e_DE_ICRS**2))

    ''' plot fake images used to detect shift '''
    fig = plt.figure('fake detected')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # apply shift:
    if False:
        import multiprocessing
        _nthreads = multiprocessing.cpu_count()
        shifted = image_registration.fft_tools.shiftnd(detected, (shift[0], shift[1]),
                                                       nthreads=_nthreads, use_numpy_fft=False)
        fig = plt.figure('fake detected with estimated offset')
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(scale_image(shifted, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    fig = plt.figure('fake reference')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(reference, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # also plot matched objects:
    pix_matched, mag_matched = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                          grid_stars=fov_stars[mask_matched], num_pix=1024,
                                          _highlight_brighter_than_mag=None, scale_bar=False,
                                          scale_bar_size=20, _display_plot=False, _save_plot=False,
                                          path='./', name='field')

    ''' solve field '''
    # measured CCD positions centered around zero:
    X = matched[:, 0:2] - (np.array(preview_img.shape) / 2.0)
    # a priori RA/Dec positions:
    Y = matched[:, 5:7]

    # initial parameters of the linear transform + distortion:
    # RA_tan, Dec_tan, x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20
    p0 = np.array([star_sc.ra.deg, star_sc.dec.deg,
                   0, 0,
                   -1. * (0.035 / 3600. * 0.999),
                   1. * (0.035 / 3600. * 0.002),
                   1. * (0.035 / 3600. * 0.002),
                   1. * (0.035 / 3600. * 0.999),
                   1e-7, 1e-7, 1e-7,
                   1e-5, 1e-7, 1e-5])

    ''' estimate linear transform parameters + 2nd order distortion '''
    # TODO: add weights depending on sextractor error?
    print('solving with LSQ')
    plsq = leastsq(residual, p0, args=(X, Y), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
    # print(plsq)
    print(plsq[0])
    print('residuals:')
    # residuals = residual(plsq[0], Y, X)
    residuals = plsq[2]['fvec']
    print(residuals)

    for jj in range(2):
        # identify outliers. they are likely to be false identifications, so discard them and redo the fit
        print('flagging outliers and refitting, take {:d}'.format(jj+1))
        # mask_outliers = residuals <= 5  # pix
        mask_outliers = np.abs(residuals) <= 9e-5  # deg
        # mask_outliers = np.abs(residuals) <= 1e-4 / (jj+1)  # deg

        # if not np.any(mask_outliers):
        #     continue

        # flag:
        X = X[mask_outliers, :]
        Y = Y[mask_outliers, :]

        plsq = leastsq(residual, plsq[0], args=(X, Y), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
        print(plsq[0])
        print('residuals:')
        residuals = plsq[2]['fvec']
        print(residuals)

        # get an estimate of the covariance matrix:
        pcov = plsq[1]
        if (len(X) > len(p0)) and pcov is not None:
            s_sq = (residuals ** 2).sum() / (len(X) - len(p0))
            pcov = pcov * s_sq
            print('covariance matrix diagonal estimate:')
            # print(pcov)
            print(pcov.diagonal())
        else:
            pcov = np.array([np.inf])

    # apply bootstrap to get a reasonable estimate of what the errors of the estimated parameters are
    print('solving with LSQ bootstrap')
    # plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, p0, X, Y, yerr_systematic=0.0, n_samp=100)
    plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, plsq[0], X, Y, yerr_systematic=0.0, n_samp=100)
    print(plsq_bootstrap)
    print(err_bootstrap)
    print('residuals:')
    residuals = residual(plsq_bootstrap, X, Y)
    print(residuals)
    print('Median residual value, mas: ', np.median(residuals)*3600*1e3)

    print('Star with smallest residual: ', X[np.argmin(residuals)], Y[np.argmin(residuals)])
    print('residual itself is {:.1f} mas'.format(residuals[np.argmin(residuals)]*3600*1e3))

    # FIXME: use bootstrapped solution:
    plsq = (plsq_bootstrap, err_bootstrap, plsq[2:])

    ''' plot the result '''
    M = np.matrix([[plsq[0][4], plsq[0][5]], [plsq[0][6], plsq[0][7]]])
    M_m1 = np.linalg.pinv(M)
    print('M:', M)
    print('M^-1:', M_m1)

    Q, R = np.linalg.qr(M)
    # print('Q:', Q)
    # print('R:', R)

    # Y_C = compute_detector_position(plsq[0], X).T + preview_img.shape[0]/2
    # Y_tan = compute_detector_position(plsq[0], np.array([list(plsq[0][0:2])])).T + preview_img.shape[0]/2
    # print(Y_C)
    # print('Tangent point pixel position: ', Y_tan)
    # print('max UV: ', compute_detector_position(plsq[0], np.array([[205.573314, 28.370672],
    #                                                                [205.564369, 28.361843]])))

    print('Estimate linear + distortion simultaneously:')
    theta = np.arccos(Q[1, 1]) * 180 / np.pi
    print('rotation angle: {:.5f} degrees'.format(theta))
    s = np.mean((R[0, 0], R[1, 1])) * 3600
    print('pixel scale: {:.7f}\" -- mean, {:.7f}\" -- x, {:.7f}\" -- y'.format(s,
                                                                               R[0, 0] * 3600,
                                                                               R[1, 1] * 3600))
    size = s * preview_img.shape[0]
    print('image size for mean pixel scale: {:.4f}\" x {:.4f}\"'.format(size, size))
    print('image size: {:.4f}\" x {:.4f}\"'.format(R[0, 0] * 3600 * preview_img.shape[0],
                                                   R[1, 1] * 3600 * preview_img.shape[1]))

    ''' plot estimated distortion map '''
    x_tan, y_tan = plsq[0][2:4]
    a_02, a_11, a_20, b_02, b_11, b_20 = plsq[0][8:]

    xy_mod_max = preview_img.shape[0] // 2
    # xy_linspace = np.linspace(-xy_mod_max, xy_mod_max, 30)
    xy_linspace = np.linspace(-xy_mod_max, xy_mod_max, 21)
    distortion_map_f = np.zeros((len(xy_linspace), len(xy_linspace)))
    distortion_map_g = np.zeros((len(xy_linspace), len(xy_linspace)))

    for i, _x in enumerate(xy_linspace):
        for j, _y in enumerate(xy_linspace):
            delta_x = _x - x_tan
            delta_y = _y - y_tan
            # uv = np.array(
            #     M * np.array([[- x_tan + a_02 * delta_y ** 2 + a_11 * delta_x * delta_y + a_20 * delta_x ** 2],
            #                   [- y_tan + b_02 * delta_y ** 2 + b_11 * delta_x * delta_y + b_20 * delta_x ** 2]]))
            # uv = np.array([[- x_tan + a_02 * delta_y ** 2 + 0*a_11 * delta_x * delta_y + 0*a_20 * delta_x ** 2],
            #                [- y_tan + b_02 * delta_y ** 2 + 0*b_11 * delta_x * delta_y + b_20 * delta_x ** 2]])
            uv = np.array([[- x_tan + a_02 * delta_y ** 2 + a_11 * delta_x * delta_y + a_20 * delta_x ** 2],
                           [- y_tan + b_02 * delta_y ** 2 + b_11 * delta_x * delta_y + b_20 * delta_x ** 2]])

            distortion_map_f[i, j] = uv[0][0]
            distortion_map_g[i, j] = uv[1][0]

    print('Median distortion in x and y:', np.median(distortion_map_f), np.median(distortion_map_g))
    print('Maximum abs distortion in x and y:', np.max(np.abs(distortion_map_f)), np.max(np.abs(distortion_map_g)))

    fig2 = plt.figure('Linear + distortion estimated simultaneously', figsize=(7, 7), dpi=120)
    ax2 = fig2.add_subplot(111)
    # single color:
    # plt.axis('equal')
    # ax2.quiver(xy_linspace, xy_linspace, distortion_map_f, distortion_map_g,  # data
    #            color=plt.cm.Blues(0.9),  # color='Teal',
    #            headlength=7)  # length of the arrows
    ax2.quiver(xy_linspace, xy_linspace, distortion_map_f, distortion_map_g,  # data
               color=plt.cm.Blues(0.9), angles='xy', scale_units='xy', scale=0.5,  # arrows are 2x magnified
               headlength=7)
    ax2.set_xlabel('x [pixels]')
    ax2.set_ylabel('y [pixels]')
    ax2.set_xlim([-640, 640])
    ax2.set_ylim([-640, 640])
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    plt.show()
