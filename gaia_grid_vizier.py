from __future__ import print_function
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import astropy.units as u
import astropy.coordinates as coord
from astropy import wcs
from astropy.io import fits
from astropy.time import Time, TimeDelta
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
import os
import ConfigParser
import inspect
from pypride.classes import inp_set, constants
from pypride.vintlib import eop_update, internet_on, load_cats, mjuliandate, taitime, eop_iers, t_eph, ter2cel, \
                            dehanttideinel, poletide, hardisp
# from pypride.vintflib import pleph

from sso_state import sso_state

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
# import seaborn as sns
import aplpy

# sns.set_style('whitegrid')
# plt.close('all')
# sns.set_context('talk')

# v = Vizier(columns=['_RAJ2000', '_DEJ2000',
#                     'RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Source', '<Gmag>'],
#            column_filters={'<Gmag>': '<22'})
v = Vizier()
v.ROW_LIMIT = -1
v.TIMEOUT = 3600


def load_sta_eop(_inp, _date, station_name='KP-VLBA'):
    const = constants()

    ''' load cats '''
    _, sta, eops = load_cats(_inp, 'DUMMY', 'S', [station_name], _date)

    ''' calculate site positions in geodetic coordinate frame
    + transformation matrix VW from VEN to the Earth-fixed coordinate frame '''
    for ii, st in enumerate(sta):
        sta[ii].geodetic(const)

    return sta, eops


def sta_compute_position(sta, eops, _date):
    """
        Get pypride station object with precomputed GCRS position for station-centric ra/decs
    :param _inp:
    :param _date: datetime object. needed to load eops?
    :param station_name:
    :return:
    """

    ''' set dates: '''
    mjd = mjuliandate(_date.year, _date.month, _date.day)
    # dd = mjd - mjd_start
    UTC = (_date.hour + _date.minute / 60.0 + _date.second / 3600.0) / 24.0
    JD = mjd + 2400000.5

    ''' compute tai & tt '''
    TAI, TT = taitime(mjd, UTC)

    ''' interpolate eops to tstamp '''
    UT1, eop_int = eop_iers(mjd, UTC, eops)

    ''' compute coordinate time fraction of CT day at 1st observing site '''
    CT, dTAIdCT = t_eph(JD, UT1, TT, sta[0].lon_gcen, sta[0].u, sta[0].v)

    ''' BCRS state vectors of celestial bodies at JD+CT, [m, m/s]: '''
    # ## Earth:
    # rrd = pleph(JD + CT, 3, 12, inp['jpl_eph'])
    # earth = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
    # # Earth's acceleration in m/s**2:
    # v_plus = np.array(pleph(JD + CT + 1.0 / 86400.0, 3, 12, inp['jpl_eph'])[3:])
    # v_minus = np.array(pleph(JD + CT - 1.0 / 86400.0, 3, 12, inp['jpl_eph'])[3:])
    # a = (v_plus - v_minus) * 1e3 / 2.0
    # a = np.array(np.matrix(a).T)
    # earth = np.hstack((earth, a))
    # ## Sun:
    # rrd = pleph(JD + CT, 11, 12, inp['jpl_eph'])
    # sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
    # ## Moon:
    # rrd = pleph(JD + CT, 10, 12, inp['jpl_eph'])
    # moon = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3

    ''' rotation matrix IERS '''
    r2000 = ter2cel(_date, eop_int, dTAIdCT, 'iau2000')

    ''' ignore displacements due to geophysical effects '''
    for ii, st in enumerate(sta):
        if st.name == 'GEOCENTR' or st.name == 'RA':
            continue
        # sta[ii] = dehanttideinel(st, _date, earth, sun, moon, r2000)
        # sta[ii] = hardisp(st, _date, r2000)
        # sta[ii] = poletide(st, _date, eop_int, r2000)

    ''' add up geophysical corrections and convert sta state to J2000 '''
    for ii, st in enumerate(sta):
        if st.name == 'GEOCENTR' or st.name == 'RA':
            continue
        sta[ii].j2000gp(r2000)

    return sta[0]


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

    # let us assume that a 6 mag star would have a flux of 10^7 counts
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
        image = convolve_fft(image, psf)

    return image


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


def radec2uv(p, x):
    """

    :param p: tangent point RA and Dec
    :param x: target RA and Dec
    :return:
    """
    if isinstance(p, list):
        p = np.array(p)
    if isinstance(x, list):
        x = np.array(x)

    # convert (ra, dec)s to 3d
    r = np.vstack((np.cos(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 1] * np.pi / 180.0))).T
    # print(r.shape)
    # print(r)

    # the same for the tangent point
    t = np.array((np.cos(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[1] * np.pi / 180.0)))
    # print(t)

    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)
    # print(u, v)

    R = r / (np.dot(r, t)[:, None])

    # print(R)

    # native tangent-plane coordinates:
    UV = 180.0 / np.pi * np.vstack((np.dot(R, u), np.dot(R, v))).T

    return UV


def uv2xy(uv, M):
    M_m1 = np.linalg.pinv(M)
    return np.dot(M_m1, uv.T)


if __name__ == '__main__':
    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(abs_path, 'config.ini'))

    _f_inp = config.get('Path', 'pypride_inp')

    # inp file for running pypride:
    inp = inp_set(_f_inp)
    inp = inp.get_section('all')

    # update pypride eops
    if internet_on():
        eop_update(inp['cat_eop'], 3)

    # asteroid database:
    path_to_database = config.get('Path', 'asteroid_database_path')
    f_database = os.path.join(path_to_database, 'ELEMENTS.NUMBR')
    # print(f_database)

    ''' main part '''

    ''' asteroid (TODO: if any? doable, but very expensive...): '''
    time_str = '20161022_042047.042560'
    exposure = TimeDelta(180, format='sec')
    start_time = Time(str(datetime.datetime.strptime(time_str, '%Y%m%d_%H%M%S.%f')), format='iso', scale='utc')
    stop_time = start_time + exposure

    asteroid_name = '3200_Phaeton'

    # load pypride stuff
    _sta, _eops = load_sta_eop(_inp=inp, _date=start_time.datetime, station_name='KP-VLBA')

    kitt_peak = sta_compute_position(sta=_sta, eops=_eops, _date=start_time.datetime)

    ra_start, dec_start, _, _, vmag_start = sso_state(_name=asteroid_name, _time=start_time,
                                                      _path_to_database=f_database,
                                                      _path_to_jpl_eph=inp['jpl_eph'],
                                                      _epoch='J2000', _station=kitt_peak)
    ra_start = Angle(ra_start)
    dec_start = Angle(dec_start)
    print(ra_start, dec_start, vmag_start)
    asteroid_start = coord.SkyCoord(ra=ra_start, dec=dec_start, frame='icrs')

    kitt_peak = sta_compute_position(sta=_sta, eops=_eops, _date=stop_time.datetime)
    ra_stop, dec_stop, _, _, vmag_stop = sso_state(_name=asteroid_name, _time=stop_time,
                                                   _path_to_database=f_database,
                                                   _path_to_jpl_eph=inp['jpl_eph'],
                                                   _epoch='J2000', _station=kitt_peak)
    print(ra_stop, dec_stop, vmag_stop)
    ra_stop = Angle(ra_stop)
    dec_stop = Angle(dec_stop)
    asteroid_stop = coord.SkyCoord(ra=ra_stop, dec=dec_stop, frame='icrs')

    ''' field '''
    field_width = Angle('60s')
    ra = Angle('22h14m53.2503s')
    dec = Angle('+52d46m47.4463s')

    catalogues = {'gaia_dr1': u'I/337/gaia', 'gsc2.3.2': u'I/305/out'}

    # target = coord.SkyCoord(ra=299.590, dec=35.201, unit=(u.deg, u.deg), frame='icrs')
    target = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')

    # guide = Vizier.query_region(target, radius=field_width, catalog=u'I/337/gaia')
    grid_stars = v.query_region(target, width=field_width, height=field_width, catalog=catalogues.values())
    # for cat in catalogues:
    #     print(grid_stars[catalogues[cat]])
    # print(grid_stars[catalogues['gaia_dr1']]['RA_ICRS'])
    # print(grid_stars[catalogues['gaia_dr1']]['_RAJ2000'])

    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = 2048
    w._naxis2 = 2048
    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2

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
    w.wcs.cd = np.array([[4.9653758578816782e-06, 7.8012027500556068e-08],
                         [8.9799574245829621e-09, 4.8009647689165968e-06]])

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

    print(w)

    # pick catalogue
    cat = catalogues['gaia_dr1']
    # cat = catalogues['gsc2.3.2']

    ''' create a [fake] simulated image '''
    # apply linear transformation only:
    # pix_stars = np.array(w.wcs_world2pix(grid_stars[cat]['_RAJ2000'], grid_stars[cat]['_DEJ2000'], 0)).T
    pix_stars = np.array(w.wcs_world2pix(grid_stars[cat]['RA_ICRS'], grid_stars[cat]['DE_ICRS'], 0)).T
    # apply linear + SIP:
    # pix_stars = np.array(w.all_world2pix(grid_stars[cat]['_RAJ2000'], grid_stars[cat]['_DEJ2000'], 0)).T
    # pix_stars = np.array(w.all_world2pix(grid_stars[cat]['RA_ICRS'], grid_stars[cat]['DE_ICRS'], 0)).T
    mag_stars = np.array(grid_stars[cat]['__Gmag_'])
    # print(pix_stars)
    # print(mag_stars)

    pix_asteroid = np.array([w.wcs_world2pix(asteroid_start.ra.deg, asteroid_start.dec.deg, 0),
                             w.wcs_world2pix(asteroid_stop.ra.deg, asteroid_stop.dec.deg, 0)])
    mag_asteroid = np.mean([vmag_start, vmag_stop])
    print(pix_asteroid)
    print(mag_asteroid)

    # load model psf for the result to look more natural
    psf_fits = '/Users/dmitryduev/_caltech/python/strehl/Strehl_calcs/SR_RESULTS/model_PSFs/lp600_scaled_SFmax.fits'
    # psf_fits = '/Users/dmitryduev/_caltech/python/strehl/Strehl_calcs/SR_RESULTS/model_PSFs/lp600_oversampled_SFmin.fits'
    try:
        with fits.open(psf_fits) as hdulist:
            model_psf = hdulist[0].data
    except IOError:
        # couldn't load? use a simple Gaussian then
        model_psf = None

    sim_image = generate_image(xy=pix_stars, mag=mag_stars,
                               xy_ast=pix_asteroid, mag_ast=mag_asteroid, exp=exposure.sec,
                               nx=w.naxis1, ny=w.naxis2, psf=model_psf)

    # convert simulated image to fits hdu:
    hdu = fits.PrimaryHDU(sim_image, header=w.to_header())

    ''' plot! '''
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

    # add asteroid 'from'->'to'
    fig.show_markers(asteroid_start.ra.deg, asteroid_start.dec.deg,
                     layer='marker_set_1', edgecolor=plt.cm.Blues(0.2),
                     facecolor=plt.cm.Blues(0.3), marker='o', s=50, alpha=0.7)
    fig.show_markers(asteroid_stop.ra.deg, asteroid_stop.dec.deg,
                     layer='marker_set_2', edgecolor=plt.cm.Oranges(0.5),
                     facecolor=plt.cm.Oranges(0.3), marker='x', s=50, alpha=0.7)

    plt.show()
