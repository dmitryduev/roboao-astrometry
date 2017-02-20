from __future__ import print_function
import sewpy
import numpy as np
import os
from skimage import exposure, img_as_float
from copy import deepcopy
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
np.set_printoptions(16)


# initialize Vizier object
with suspend_cache(Vizier):
    viz = Vizier()
    viz.ROW_LIMIT = -1
    viz.TIMEOUT = 30


# import operator as op
# def ncr(n, r):
#     r = min(r, n-r)
#     if r == 0: return 1
#     numer = reduce(op.mul, range(n, n-r, -1))
#     denom = reduce(op.mul, range(1, r+1))
#     return numer//denom


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
        image = gaussian_filter(image, 7)
    else:
        # convolve with a (model) psf
        # fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(nthreads=4, use_numpy_fft=False)
        # image = convolve_fft(image, psf, fftn=fftn, ifftn=ifftn)
        image = convolve_fft(image, psf)

    return image


def make_image(target, window_size, _model_psf, pix_stars, mag_stars):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = int(1024 * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    w._naxis2 = int(1024 * (window_size[1] * 180.0 / np.pi * 3600) / 36)
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
    w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
                         [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2

    ''' create a [fake] simulated image '''
    # tic = _time()
    sim_image = generate_image(xy=pix_stars, mag=mag_stars, nx=w.naxis1, ny=w.naxis2, psf=_model_psf)

    return sim_image


def plot_field(target, window_size, _model_psf, grid_stars, _highlight_brighter_than_mag=16.0,
               scale_bar=False, scale_bar_size=20, _display_plot=False, _save_plot=False, path='./', name='field'):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    # w._naxis1 = int(2048 * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    # w._naxis2 = int(2048 * (window_size[1] * 180.0 / np.pi * 3600) / 36)
    w._naxis1 = int(1024 * (window_size[0] * 180.0 / np.pi * 3600) / 36)
    w._naxis2 = int(1024 * (window_size[1] * 180.0 / np.pi * 3600) / 36)
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
    w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
                         [8.9799574245829621e-09, 4.8009647689165968e-06]]) * 2
    # w.wcs.cd = np.array([[5e-06, 0],
    #                      [0, 5e-06]]) * 2

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
    # mask_bright = mag_stars <= _highlight_brighter_than_mag
    # if np.max(mask_bright) == 1:
    #     fig.show_markers(grid_stars[mask_bright]['RA_ICRS'], grid_stars[mask_bright]['DE_ICRS'],
    #                      layer='marker_set_2', edgecolor=plt.cm.Oranges(0.9),
    #                      facecolor=plt.cm.Oranges(0.8), marker='+', s=50, alpha=0.9, linewidths=1)

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


def triangulate(xy, cut=None, max_pix=None):

    if cut is not None:
        xy_brightest = xy[:cut, :]

    quads = []
    for q in combinations(xy_brightest, 4):
        # print(q)
        len_max_edge = 0
        max_edge = None
        for duplet in combinations(q, 2):
            len_edge = np.linalg.norm(duplet[1]-duplet[0])
            if len_edge > len_max_edge:
                len_max_edge = len_edge
                max_edge = duplet

        if max_pix is not None and len_max_edge > max_pix:
            continue

        hash = []
        for p in q:
            p_len = np.linalg.norm(p - max_edge[0])
            if p_len != 0:
                # compute projections of the 'inner' points and store that as a hash for the quad:
                cos = np.dot(p - max_edge[0], max_edge[1] - max_edge[0]) / p_len / len_max_edge
                proj_hash = p_len * cos / len_max_edge

                if proj_hash != 1.0:
                    hash.append(proj_hash)

        quads.append([q, hash])

    return quads


if __name__ == '__main__':
    path_in = '/Users/dmitryduev/_caltech/roboao/_faint_reductions/20170211/0_M13_VIC_Si_o_20170211_122715.043747/'
    # fits_in = '100p.fits'
    fits_in = '0_M13_VIC_Si_o_20170211_122715.043747_blind_decnv.fits'

    # see /usr/local/Cellar/sextractor/2.19.5/share/sextractor/default.sex

    # sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "XWIN_IMAGE", "YWIN_IMAGE",
    #                         "ERRAWIN_IMAGE", "ERRBWIN_IMAGE", "ERRTHETAWIN_IMAGE",
    #                         "FLUX_AUTO", "FLUXERR_AUTO",
    #                         "A_IMAGE", "B_IMAGE", "FWHM_IMAGE",
    #                         "FLAGS", "FLAGS_WEIGHT", "FLUX_RADIUS"],
    #                 config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '4.0'},
    #                 sexpath="sex")
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

    # generate triangles:
    pix_det = np.vstack((out['table']['X_IMAGE'], out['table']['Y_IMAGE'])).T
    pix_det_err = np.vstack((out['table']['X2_IMAGE'], out['table']['Y2_IMAGE'], out['table']['XY_IMAGE'])).T
    mag_det = np.array(out['table']['FLUX_AUTO'])
    # brightest 20:
    # tic = _time()
    # quads_detected = triangulate(xy, cut=20)
    # print(_time() - tic)

    # for l in out['table']:
    #     print(l)
    print(out['table'])  # This is an astropy table.
    # print('detected {:d} sources'.format(len(out['table'])))
    # print(np.mean(out['table']['A_IMAGE']), np.mean(out['table']['B_IMAGE']))
    # print(np.median(out['table']['A_IMAGE']), np.median(out['table']['B_IMAGE']))

    # load first image frame from the fits file
    preview_img, header = load_fits(os.path.join(path_in, fits_in), return_header=True)
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

    ''' get stats from catalogue, create fake images, then cross correlate them '''
    # stars in the field (without mag cut-off):
    # print(header['OBJRA'][0], header['OBJDEC'][0])
    star_sc = SkyCoord(ra=header['OBJRA'][0], dec=header['OBJDEC'][0], frame='icrs')

    # search radius: " -> rad
    fov_size_ref = 100 * np.pi / 180.0 / 3600

    catalog = 'I/337/gaia'
    stars_table = viz.query_region(star_sc, width=fov_size_ref * u.rad,
                                   height=fov_size_ref * u.rad,
                                   catalog=catalog, cache=False)
    fov_stars = stars_table[catalog]

    pix_ref, mag_ref = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                  grid_stars=fov_stars, _highlight_brighter_than_mag=16.0, scale_bar=False,
                                  scale_bar_size=20, _display_plot=False, _save_plot=False, path='./', name='field')

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
    naxis_det = int(1024 * (fov_size_det * 180.0 / np.pi * 3600) / 36)
    naxis_ref = int(1024 * (fov_size_ref * 180.0 / np.pi * 3600) / 36)
    # effectively shift detected positions to center of ref frame to reduce distortion effect
    pix_det_ref = pix_det + np.array([naxis_ref//2, naxis_ref//2]) - np.array([naxis_det//2, naxis_det//2])
    detected = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                          pix_stars=pix_det_ref, mag_stars=mag_det)
    reference = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                           pix_stars=pix_ref, mag_stars=mag_ref)

    # register shift: pixel precision first
    from skimage.feature import register_translation
    shift, error, diffphase = register_translation(reference, detected)
    print(shift, error)

    # associate!
    matched = []
    mask_matched = []
    for si, s in enumerate(pix_det_ref):
        s_shifted = s + np.array(shift[::-1])

        pix_distance = np.min(np.linalg.norm(pix_ref - s_shifted, axis=1))
        print(pix_distance)

        if pix_distance < 10:
            min_ind = np.argmin(np.linalg.norm(pix_ref - s_shifted, axis=1))

            matched.append(np.hstack([pix_det[si], pix_det_err[si],
                                        np.array([fov_stars['RA_ICRS'][min_ind],
                                                  fov_stars['DE_ICRS'][min_ind],
                                                  fov_stars['e_RA_ICRS'][min_ind],
                                                  fov_stars['e_DE_ICRS'][min_ind],
                                                  fov_stars['RADEcor'][min_ind]])]))
            mask_matched.append(min_ind)

    matched = np.array(matched)
    # print(matched)
    print('total matched:', len(matched))

    ''' plot fake images used to detect shift '''
    fig = plt.figure('fake detected')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    fig = plt.figure('fake reference')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(reference, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # also plot matched objects:
    pix_matched, mag_matched = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                  grid_stars=fov_stars[mask_matched], _highlight_brighter_than_mag=16.0, scale_bar=False,
                                  scale_bar_size=20, _display_plot=True, _save_plot=False, path='./', name='field')

    ''' solve field '''
    import pyBA
    objectsA = np.array([pyBA.Bivarg(mu=ix[0:2], sigma=ix[2:5]) for ix in matched])
    objectsB = np.array([pyBA.Bivarg(mu=ix[5:7], sigma=ix[7:10]) for ix in matched])

    S = pyBA.background.suggest_mapping(objectsA, objectsB)
    print(S.mu)
    # print(S.sigma)

    # Get maximum a posteriori background mapping parameters
    P = pyBA.background.MAP(objectsA, objectsB, mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True)
    print(P.mu)

    # Create astrometric mapping object
    D = pyBA.Amap(P, objectsA, objectsB)
    D.scale = 100
    nres = 30  # Density of interpolation grid points

    # plt.show()
