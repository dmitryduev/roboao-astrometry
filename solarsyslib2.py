from __future__ import print_function
import numpy as np
import os
import datetime
import urllib2
import ConfigParser
import inspect
from pypride.vintflib import pleph
from pypride.classes import inp_set, constants
from pypride.vintlib import eop_update, internet_on, load_cats, mjuliandate, taitime, eop_iers, t_eph, ter2cel
from pypride.vintlib import sph2cart, cart2sph, iau_PNM00A
from pypride.vintlib import factorise, aber_source, R_123
from astroplan import Observer as Observer_astroplan
# from astroplan import FixedTarget
from astroplan import observability_table  # , is_observable, is_always_observable
# from astroplan.plots import plot_sky
from astroplan import time_grid_from_range
from astroplan import AtNightConstraint, AltitudeConstraint
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from time import time as _time
import multiprocessing
import pytz
from numba import jit
from copy import deepcopy
import traceback
from astroquery.vizier import Vizier

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import aplpy


# initialize Vizier object
viz = Vizier()
viz.ROW_LIMIT = -1
viz.TIMEOUT = 3600


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


@jit
def dms(rad):
    d, m = divmod(abs(rad), np.pi/180)
    m, s = divmod(m, np.pi/180/60)
    s /= np.pi/180/3600
    if rad >= 0:
        return [d, m, s]
    else:
        return [-d, -m, -s]


@jit
def hms(rad):
    if rad < 0:
        rad += np.pi
    h, m = divmod(rad, np.pi/12)
    m, s = divmod(m, np.pi/12/60)
    s /= np.pi/12/3600
    return [h, m, s]


def _target_is_vector(target):
    if hasattr(target, '__iter__'):
        return True
    else:
        return False


def target_list_all_helper(args):
    """ Helper function to run asteroid computation in parallel

    :param args:
    :return:
    """
    targlist, asteroid, mjd, night = args
    radec, radec_rate, Vmag = targlist.getObsParams(asteroid, mjd)
    # meridian_transit = targlist.get_hour_angle_limit(night, radec[0], radec[1])
    # return [radec, radec_rate, Vmag, meridian_transit]
    return [radec, radec_rate, Vmag]


def hour_angle_limit_helper(args):
    """ Helper function to run hour angle limit computation in parallel

    :param args:
    :return:
    """
    targlist, radec, night = args
    meridian_transit, t_azel = targlist.get_hour_angle_limit2(night, radec[0], radec[1], N=20)
    return [meridian_transit, t_azel]


# overload target meridian transit
class Observer(Observer_astroplan):

    def _determine_which_event(self, function, args_dict):
        """
        Run through the next/previous/nearest permutations of the solutions
        to `function(time, ...)`, and return the previous/next/nearest one
        specified by the args stored in args_dict.
        """
        time = args_dict.pop('time', None)
        target = args_dict.pop('target', None)
        which = args_dict.pop('which', None)
        horizon = args_dict.pop('horizon', None)
        rise_set = args_dict.pop('rise_set', None)
        antitransit = args_dict.pop('antitransit', None)
        N = 20

        # Assemble arguments for function, depending on the function.
        if function == self._calc_riseset:
            args = lambda w: (time, target, w, rise_set, horizon, N)
        elif function == self._calc_transit:
            args = lambda w: (time, target, w, antitransit, N)
        else:
            raise ValueError('Function {} not supported in '
                             '_determine_which_event.'.format(function))

        if not isinstance(time, Time):
            time = Time(time)

        if which == 'next' or which == 'nearest':
            next_event = function(*args('next'))
            if which == 'next':
                return next_event

        if which == 'previous' or which == 'nearest':
            previous_event = function(*args('previous'))
            if which == 'previous':
                return previous_event

        if which == 'nearest':
            if _target_is_vector(target):
                return_times = []
                for next_e, prev_e in zip(next_event, previous_event):
                    if abs(time - prev_e) < abs(time - next_e):
                        return_times.append(prev_e)
                    else:
                        return_times.append(next_e)
                return Time(return_times)
            else:
                if abs(time - previous_event) < abs(time - next_event):
                    return previous_event
                else:
                    return next_event


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


class Target(object):
    """
        Store data for a single object at epoch
    """
    def __init__(self, _object):
        self.object = _object
        # position at the middle of the night:
        self.epoch = None
        self.radec = None
        self.radec_dot = None
        self.mag = None
        self.meridian_crossing = None
        self.is_observable = None
        self.t_el = None
        self.observability_windows = None
        self.guide_stars = None

    def __str__(self):
        """
            Print it out nicely
        """
        out_str = '<Target object: {:s} '.format(str(self.object))
        if self.epoch is not None:
            out_str += '\n epoch: {:s}'.format(str(self.epoch))
        if self.mag is not None:
            out_str += '\n magnitude: {:.3f}'.format(self.mag)
        if self.radec is not None:
            out_str += '\n ra: {:.10f} rad, dec: {:.10f} rad'.format(*self.radec)
        if self.radec_dot is not None:
            out_str += '\n ra_dot: {:.3f} arcsec/sec, dec_dot: {:.3f} arcsec/sec'.format(*self.radec_dot)
        if self.meridian_crossing is not None:
            out_str += '\n meridian crossing: {:s}'.format(str(self.meridian_crossing))
        if self.is_observable is not None:
            out_str += '\n is observable: {:s}'.format(str(self.is_observable))
        # if self.t_el is not None:
        #     out_str += '\n time elv: {:s}'.format(str(self.t_el))
        if self.observability_windows is not None:
            out_str += '\n observability windows: {:s}'.format(str(self.observability_windows))
        if self.guide_stars is not None:
            out_str += '\n guide stars: {:s}'.format(np.array(self.guide_stars))
        return out_str + '>'

    def set_epoch(self, _epoch):
        """
            Position at the middle of the night
        :param _epoch:
        :return:
        """
        self.epoch = _epoch
        # reset everything else if epoch is changed
        self.radec = None
        self.radec_dot = None
        self.mag = None
        self.meridian_crossing = None
        self.is_observable = None
        self.t_el = None
        self.observability_windows = None
        self.guide_stars = None

    def set_radec(self, _radec):
        self.radec = _radec

    def set_radec_dot(self, _radec_dot):
        self.radec_dot = _radec_dot

    def set_mag(self, _mag):
        self.mag = _mag

    def set_meridian_crossing(self, _meridian_crossing):
        self.meridian_crossing = _meridian_crossing

    def set_is_observable(self, _is_observable):
        self.is_observable = _is_observable

    def set_t_el(self, _t_el):
        self.t_el = _t_el

    def set_observability_windows(self, _elv_lim=45.0):

        # get elv vs time:
        t_el = np.array(self.t_el)
        # print(t_el)
        # print(t_el.shape)
        N = t_el.shape[0]
        t = np.linspace(0, 1, N)
        # fit elv vs time with a poly
        p = np.polyfit(t, map(float, t_el[:, 1]), 6)
        # evaluate it on a denser grid to estimate periods when elv >= min_elv more precisely
        N_dense = 200
        t_dense = np.linspace(0, 1, N_dense)
        dense = np.polyval(p, t_dense)

        scans = []
        scan = []
        # print(self.elv_lim)
        for t_d, el in zip(t_dense, dense):
            # print(t_d, el)
            if el >= _elv_lim:
                scan.append(t_d)
                # print('appended ', t_d, ' for ', el)
            else:
                if len(scan) > 1:
                    # print('scan ended: ', [scan[0], scan[-1]])
                    scans.append([scan[0], scan[-1]])
                scan = []
        # append last scan:
        if len(scan) > 1:
            # print('scan ended: ', [scan[0], scan[-1]])
            scans.append([scan[0], scan[-1]])
        # convert to Time objects:
        if len(scans) > 0:
            t_0 = Time(t_el[0, 0], format='iso')
            t_e = Time(t_el[-1, 0], format='iso')
            dt = t_e - t_0
            scans = t_0 + scans * dt

        self.observability_windows = scans

    def set_guide_stars(self, _jpl_eph, _guide_star_cat=u'I/337/gaia', _station=None,
                        _radius=30.0, _margin=30.0, _m_lim_gs=16.0, _plot_field=False,
                        _model_psf=None):
        """
            Get guide stars within radius arcseconds for each observability window.
        :param _jpl_eph:
        :param _station:
        :param _radius: maximum distance to guide star in arcseconds
        :param _margin: padd window with margin arcsec (one-sided margin)
        :return:
        """
        if self.observability_windows is None:
            print('compute observability windows first before looking for guide stars')
            return
        elif len(self.observability_windows) == 0 or not self.is_observable:
            print('target {:s} not observable'.format(self.object.name))
            return
        else:
            print('target {:s} observable:'.format(self.object.name))
            # init list to keep guide stars
            self.guide_stars = []
            # search radius: " -> rad
            radius_rad = _radius * np.pi / 180.0 / 3600

        # print(self.object.name)
        for window in self.observability_windows:
            # start of the 'arc'
            t_start = window[0]
            radec_start, _, vmag_start = self.object.raDecVmag(t_start.mjd, _jpl_eph, epoch='J2000',
                                                               station=_station, output_Vmag=True)
            radec_start = np.array(radec_start)
            # end of the 'arc'
            t_stop = window[1]
            radec_stop, _, vmag_stop = self.object.raDecVmag(t_stop.mjd, _jpl_eph, epoch='J2000',
                                                             station=_station, output_Vmag=True)
            radec_stop = np.array(radec_stop)
            # middle point for the 'FoV':
            radec_middle = (radec_start + radec_stop) / 2.0
            # 'FoV' size + margins at both sides:
            window_size = np.abs(radec_stop - radec_start) + 2.0*np.array([_margin*np.pi/180.0/3600,
                                                                           _margin*np.pi/180.0/3600])
            # in arcsec:
            # print(window_size*180.0/np.pi*3600)

            # window time span
            window_t_span = t_stop - t_start

            target = coord.SkyCoord(ra=radec_middle[0], dec=radec_middle[1], unit=(u.rad, u.rad), frame='icrs')
            global viz
            # viz.column_filters = {'<Gmag>': '<{:.1f}'.format(_m_lim_gs)}
            grid_stars = viz.query_region(target, width=window_size[0]*u.rad, height=window_size[1]*u.rad,
                                          catalog=_guide_star_cat)
            # print(guide_stars[_guide_star_cat])
            # guide star magnitudes:
            grid_star_mags = np.array(grid_stars[_guide_star_cat]['__Gmag_'])
            # those that are bright enough for tip-tilt:
            mag_mask = grid_star_mags <= _m_lim_gs

            if _plot_field:
                self.plot_field(target, window_size=window_size,
                                radec_start=radec_start, vmag_start=vmag_start,
                                radec_stop=radec_stop, vmag_stop=vmag_stop,
                                exposure=t_stop-t_start,
                                _model_psf=_model_psf, grid_stars=grid_stars[_guide_star_cat])

            # compute distances from stars, return those (bright ones) that can be used as guide stars
            for star in grid_stars[_guide_star_cat][mag_mask]:
                radec_star = np.array([star['RA_ICRS'], star['DE_ICRS']]) * np.pi / 180.0
                # print(star)
                print(radec_star)
                print(radec_start, radec_stop)
                '''
                # Consider the line extending the segment, parameterized as v + t (w - v).
                # We find projection of point p onto the line.
                # It falls where t = [(p-v) . (w-v)] / |w-v|^2
                # We clamp t from [0,1] to handle points outside the segment vw.
                '''
                vw = radec_stop - radec_start
                l = np.linalg.norm(vw)
                l2 = l ** 2
                t = np.max([0, np.min([1, np.dot(radec_star - radec_start, vw) / l2])])
                projection = radec_start + t * vw

                distance = np.linalg.norm(radec_star - projection)

                # not too far away?
                if radius_rad >= distance:

                    window_border = np.sqrt(radius_rad ** 2 - distance ** 2)
                    delta_t_start = np.max([0, t - window_border / l])
                    # start (linear) RA/Dec position
                    # window_start = radec_start + delta_t_start * vw
                    delta_t_stop = np.min([1, t + window_border / l])
                    # stop (linear) RA/Dec position
                    # window_stop = radec_start + delta_t_stop * vw
                    # print(window_start, window_stop)
                    # delta_t_'s must be multiplied by the full streak time span to get the plausible time range:
                    # print(delta_t_start, delta_t_stop)
                    t_start_star = t_start + delta_t_start * window_t_span
                    t_stop_star = t_start + delta_t_stop * window_t_span
                    # save:
                    self.guide_stars.append([star, distance, [t_start_star, t_stop_star]])
                # else:
                #     print('too far away')

    @staticmethod
    def plot_field(target, window_size, radec_start, vmag_start, radec_stop, vmag_stop,
                   exposure, _model_psf, grid_stars):
        """

        :return:
        """
        ''' set up WCS '''
        # Create a new WCS object.  The number of axes must be set
        # from the start
        w = wcs.WCS(naxis=2)
        w._naxis1 = int(2048 * (window_size[0]*180.0/np.pi*3600) / 36)
        w._naxis2 = int(2048 * (window_size[1]*180.0/np.pi*3600) / 36)
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

        ''' create a [fake] simulated image '''
        # apply linear transformation only:
        pix_stars = np.array(w.wcs_world2pix(grid_stars['RA_ICRS'], grid_stars['DE_ICRS'], 0)).T
        # apply linear + SIP:
        # pix_stars = np.array(w.all_world2pix(grid_stars['_RAJ2000'], grid_stars['_DEJ2000'], 0)).T
        # pix_stars = np.array(w.all_world2pix(grid_stars['RA_ICRS'], grid_stars['DE_ICRS'], 0)).T
        mag_stars = np.array(grid_stars['__Gmag_'])
        # print(pix_stars)
        # print(mag_stars)

        asteroid_start = coord.SkyCoord(ra=radec_start[0], dec=radec_start[1], unit=(u.rad, u.rad), frame='icrs')
        asteroid_stop = coord.SkyCoord(ra=radec_stop[0], dec=radec_stop[1], unit=(u.rad, u.rad), frame='icrs')

        pix_asteroid = np.array([w.wcs_world2pix(asteroid_start.ra.deg, asteroid_start.dec.deg, 0),
                                 w.wcs_world2pix(asteroid_stop.ra.deg, asteroid_stop.dec.deg, 0)])
        mag_asteroid = np.mean([vmag_start, vmag_stop])
        # print(pix_asteroid)
        # print(mag_asteroid)

        sim_image = generate_image(xy=pix_stars, mag=mag_stars,
                                   xy_ast=pix_asteroid, mag_ast=mag_asteroid, exp=exposure.sec,
                                   nx=w.naxis1, ny=w.naxis2, psf=_model_psf)

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


class TargetListAsteroids(object):
    """
        Produce (nightly) target list for the asteroids project
    """

    def __init__(self, _f_inp, database_source='mpc', database_file=None, _observatory='kitt peak',
                 _m_lim=16.0, _elv_lim=40.0, date=None, timezone='America/Phoenix'):
        # init database
        if database_source == 'mpc':
            if database_file is None:
                db = AsteroidDatabaseMPC()
            else:
                db = AsteroidDatabaseMPC(_f_database=database_file)
        elif database_source == 'jpl':
            if database_file is None:
                db = AsteroidDatabaseJPL()
            else:
                db = AsteroidDatabaseJPL(_f_database=database_file)
        else:
            raise Exception('database source not understood')
        # load the database
        db.load()

        self.database = db.database
        # observatory object
        self.observatory = Observer.at_site(_observatory)

        # minimum object magnitude to be output
        self.m_lim = _m_lim

        # elevation cutoff
        self.elv_lim = _elv_lim

        # inp file for running pypride:
        inp = inp_set(_f_inp)
        self.inp = inp.get_section('all')

        # update pypride eops
        if internet_on():
            eop_update(self.inp['cat_eop'], 3)

        ''' load eops '''
        if date is None:
            now = datetime.datetime.now(pytz.timezone(timezone))
            date = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

        # load pypride stuff
        if _observatory == 'kpno':
            _sta, self.eops = load_sta_eop(_inp=self.inp, _date=date, station_name='KP-VLBA')
        else:
            print('Station is not Kitt Peak! Falling back to geocentric ra/decs')
            _sta, self.eops = load_sta_eop(_inp=self.inp, _date=date, station_name='GEOCENTR')

        self.observatory_pypride = sta_compute_position(sta=_sta, eops=self.eops, _date=date)

        ''' precalc vw matrix '''
        lat = self.observatory.location.latitude.rad
        lon = self.observatory.location.longitude.rad
        # Compute the local VEN-to-crust-fixed rotation matrices by rotating
        # about the geodetic latitude and the longitude.
        # w - rotation matrix by an angle lat_geod around the y axis
        w = R_123(2, lat)
        # v - rotation matrix by an angle -lon_gcen around the z axis
        v = R_123(3, -lon)
        # product of the two matrices:
        self.vw = np.dot(v, w)

        ''' targets '''
        self.targets = np.array([], dtype=object)

    def get_hour_angle_limit(self, night, ra, dec):
        # calculate meridian transit time to set hour angle limit.
        # no need to observe a planet when it's low if can wait until it's high.
        # get next transit after night start:
        meridian_transit_time = self.observatory.target_meridian_transit_time(night[0],
                                                                              SkyCoord(ra=ra * u.rad, dec=dec * u.rad),
                                                                              which='next')

        # will it happen during the night?
        # print('astroplan ', meridian_transit_time.iso)
        meridian_transit = (night[0] <= meridian_transit_time <= night[1])

        return meridian_transit

    @staticmethod
    def _generate_24hr_grid(t0, start, end, N, for_deriv=False):
        """
        Generate a nearly linearly spaced grid of time durations.
        The midpoints of these grid points will span times from ``t0``+``start``
        to ``t0``+``end``, including the end points, which is useful when taking
        numerical derivatives.
        Parameters
        ----------
        t0 : `~astropy.time.Time`
            Time queried for, grid will be built from or up to this time.
        start : float
            Number of days before/after ``t0`` to start the grid.
        end : float
            Number of days before/after ``t0`` to end the grid.
        N : int
            Number of grid points to generate
        for_deriv : bool
            Generate time series for taking numerical derivative (modify
            bounds)?
        Returns
        -------
        `~astropy.time.Time`
        """

        if for_deriv:
            time_grid = np.concatenate([[start - 1 / (N - 1)],
                                        np.linspace(start, end, N)[1:-1],
                                        [end + 1 / (N - 1)]]) * u.day
        else:
            time_grid = np.linspace(start, end, N) * u.day

        return t0 + time_grid

    @staticmethod
    def altaz(tt, eops, jpl_eph, vw, r_GTRS, ra, dec):
        """
        """
        ''' set coordinates '''
        K_s = np.array([np.cos(dec) * np.cos(ra),
                        np.cos(dec) * np.sin(ra),
                        np.sin(dec)])

        azels = []

        for t in tt:
            ''' set dates: '''
            tstamp = t.datetime
            mjd = np.floor(t.mjd)
            UTC = (tstamp.hour + tstamp.minute / 60.0 + tstamp.second / 3600.0) / 24.0
            JD = mjd + 2400000.5

            ''' compute tai & tt '''
            TAI, TT = taitime(mjd, UTC)

            ''' interpolate eops to tstamp '''
            UT1, eop_int = eop_iers(mjd, UTC, eops)

            ''' compute coordinate time fraction of CT day at GC '''
            CT, dTAIdCT = t_eph(JD, UT1, TT, 0.0, 0.0, 0.0)

            ''' rotation matrix IERS '''
            r2000 = ter2cel(tstamp, eop_int, dTAIdCT, 'iau2000')

            ''' do not compute displacements due to geophysical effects '''

            ''' get only the velocity '''
            v_GCRS = np.dot(r2000[:, :, 1], r_GTRS)

            ''' earth position '''
            rrd = pleph(JD + CT, 3, 12, jpl_eph)
            earth = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3

            az, el = aber_source(v_GCRS, vw, K_s, r2000, earth)
            azels.append([az, el])

        return azels

    def get_hour_angle_limit2(self, time, ra, dec, N=20):
        """
        Time at next transit of the meridian of `target`.
        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)
        ra : object RA
        dec : object Dec
        N : int
            Number of altitudes to compute when searching for
            rise or set.
        Returns
        -------
        ret1 : bool if target crosses the meridian or not during the night
        """
        if not isinstance(time, Time):
            time = Time(time)
        times = self._generate_24hr_grid(time[0], 0, 1, N, for_deriv=False)

        # The derivative of the altitude with respect to time is increasing
        # from negative to positive values at the anti-transit of the meridian
        # if antitransit:
        #     rise_set = 'rising'
        # else:
        #     rise_set = 'setting'

        r_GTRS = np.array(self.observatory.location.value)

        altaz = self.altaz(times, self.eops, self.inp['jpl_eph'], self.vw, r_GTRS, ra, dec)
        altitudes = np.array(altaz)[:, 1]
        # print(zip(times.iso, altitudes*180/np.pi))
        # print('\n')

        t = np.linspace(0, 1, N)
        p = np.polyfit(t, altitudes, 6)
        dense = np.polyval(p, np.linspace(0, 1, 200))
        maxp = np.max(dense)
        root = np.argmax(dense)/200.0
        minus = np.polyval(p, maxp - 0.01)
        plus = np.polyval(p, maxp + 0.01)

        # print('pypride ', (time[0] + root * u.day).iso)
        # return bool if crosses the meridian and array with elevations
        return (time[0] + root*u.day < time[1]) and (np.max((minus, plus)) < maxp), \
                zip(times.iso, altitudes*180/np.pi)

    def target_list_all(self, day, mask=None, parallel=False):
        """
            Get observational parameters for a (masked) target list
            from self.database
        """
        # get middle of night:
        night, middle_of_night = self.middle_of_night(day)
        mjd = middle_of_night.tdb.mjd  # in TDB!!
        # print(middle_of_night.datetime)
        # iterate over asteroids:
        target_list = []
        if mask is None:
            mask = range(0, len(self.database))

        # do it in parallel
        if parallel:
            ttic = _time()
            n_cpu = multiprocessing.cpu_count()
            # create pool
            pool = multiprocessing.Pool(n_cpu)
            # asynchronously apply target_list_all_helper
            database_masked = self.database[mask]
            args = [(self, asteroid, mjd, night) for asteroid in database_masked]
            result = pool.map_async(target_list_all_helper, args)
            # close bassejn
            pool.close()  # we are not adding any more processes
            pool.join()  # wait until all threads are done before going on
            # get the ordered results
            targets_all = result.get()
            # print(targets_all)
            # get only the asteroids that are bright enough
            target_list = [[AsteroidDatabase.init_asteroid(database_masked[_it])] + [middle_of_night] + _t
                            for _it, _t in enumerate(targets_all) if _t[2] <= self.m_lim]

            # set hour angle limit if asteroid crosses the meridian during the night
            pool = multiprocessing.Pool(n_cpu)
            # asynchronously apply target_list_all_helper
            target_list = np.array(target_list)
            args = [(self, radec, night) for (_, _, radec, _, _) in target_list]
            result = pool.map_async(hour_angle_limit_helper, args)
            # close bassejn
            pool.close()  # we are not adding any more processes
            pool.join()  # wait until all threads are done before going on
            # get the ordered results
            meridian_transiting_asteroids = np.array(result.get(), dtype=object)
            # stack the result with target_list
            target_list = np.hstack((target_list, meridian_transiting_asteroids))
            print('parallel computation took: {:.2f} s'.format(_time() - ttic))
        else:
            ttic = _time()
            for ia, asteroid in enumerate(self.database[mask]):
                # print '\n', len(self.database)-ia
                # print('\n', asteroid)
                # in the middle of night...
                # tic = _time()
                radec, radec_dot, Vmag = self.getObsParams(asteroid, mjd)
                # print(len(self.database)-ia, _time() - tic)
                # skip if too dim
                if Vmag <= self.m_lim:
                    # ticcc = _time()
                    # meridian_transit = self.get_hour_angle_limit(night, radec[0], radec[1])
                    # print(_time() - ticcc, meridian_transit)
                    # ticcc = _time()
                    meridian_transit, t_azel = self.get_hour_angle_limit2(night, radec[0], radec[1], N=20)
                    # print(_time() - ticcc, meridian_transit)
                    target_list.append([AsteroidDatabase.init_asteroid(asteroid), middle_of_night,
                                        radec, radec_dot, Vmag, meridian_transit, t_azel])
            target_list = np.array(target_list)
            print('serial computation took: {:.2f} s'.format(_time() - ttic))
        # print('Total targets brighter than 16.5', len(target_list))

        # set up target objects for self.targets:
        for entry in target_list:
            target = Target(_object=entry[0])
            target.set_epoch(entry[1])
            target.set_radec(entry[2])
            target.set_radec_dot(entry[3])
            target.set_mag(entry[4])
            target.set_meridian_crossing(entry[5])
            target.set_t_el(entry[6])
            self.targets = np.append(self.targets, target)
        # print(self.targets)

    def target_list_observable(self, day, twilight='nautical', fraction=0.1):
        """ Check whether targets are observable and return only those

        :param day:
        :param twilight:
        :param fraction:
        :return:
        """
        if self.targets is None:
            print('no targets in the target list')
            return

        night, middle_of_night = self.middle_of_night(day)
        # set constraints (above self.elv_lim deg altitude, Sun altitude < -N deg [dep.on twilight])
        constraints = [AltitudeConstraint(self.elv_lim * u.deg, 90 * u.deg)]
        if twilight == 'nautical':
            constraints.append(AtNightConstraint.twilight_nautical())
        elif twilight == 'astronomical':
            constraints.append(AtNightConstraint.twilight_astronomical())
        elif twilight == 'civil':
            constraints.append(AtNightConstraint.twilight_civil())

        radec = np.array([_target.radec for _target in self.targets])
        # print(radec)
        # tic = _time()
        coords = SkyCoord(ra=radec[:, 0], dec=radec[:, 1],
                          unit=(u.rad, u.rad), frame='icrs')
        # print(_time() - tic)
        tic = _time()
        table = observability_table(constraints, self.observatory, coords,
                                    time_range=night)
        print('observability computation took: {:.2f} s'.format(_time() - tic))
        # print(table)

        # proceed with observable (for more than 5% of the night) targets only
        mask_observable = table['fraction of time observable'] > fraction
        print(mask_observable)

        target_list_observeable = self.targets[mask_observable]
        print('total bright asteroids: ', len(self.targets),
              'observable: ', len(target_list_observeable))

        # self.targets = target_list_observeable
        for target in self.targets[mask_observable]:
            target.set_is_observable(True)
        for target in self.targets[~mask_observable]:
            target.set_is_observable(False)

    def getObsParams(self, target, mjd):
        """ Compute obs parameters for a given t

        :param target: Kepler class object
        :param mjd: epoch in TDB/mjd (t.tdb.mjd, t - astropy.Time object, UTC)
        :return: radec in rad, radec_dot in arcsec/s, Vmag
        """

        AU_DE421 = 1.49597870699626200e+11  # m
        GSUN = 0.295912208285591100e-03 * AU_DE421**3 / 86400.0**2
        # convert AU to m:
        a = target['a'] * AU_DE421
        e = target['e']
        # convert deg to rad:
        i = target['i'] * np.pi / 180.0
        w = target['w'] * np.pi / 180.0
        Node = target['Node'] * np.pi / 180.0
        M0 = target['M0'] * np.pi / 180.0
        t0 = target['epoch']
        H = target['H']
        G = target['G']

        asteroid = Asteroid(target['name'], a, e, i, w, Node, M0, GSUN, t0, H, G)

        # jpl_eph - path to eph used by pypride
        radec, radec_dot, Vmag = asteroid.raDecVmag(mjd, self.inp['jpl_eph'], epoch='J2000',
                                                    station=self.observatory_pypride, output_Vmag=True)
        #    print(radec.ra.hms, radec.dec.dms, radec_dot, Vmag)

        return radec, radec_dot, Vmag

    def middle_of_night(self, day):
        """
            day - datetime.datetime object, 0h UTC of the coming day
        """
        #        day = datetime.datetime(2015,11,7) # for KP, in UTC it is always 'tomorrow'
        nextDay = day + datetime.timedelta(days=1)
        astrot = Time([str(day), str(nextDay)], format='iso', scale='utc')
        # when the night comes, heh?
        sunSet = self.observatory.sun_set_time(astrot[0])
        sunRise = self.observatory.sun_rise_time(astrot[1])

        night = Time([str(sunSet.datetime), str(sunRise.datetime)],
                     format='iso', scale='utc')

        # build time grid for the night to come
        time_grid = time_grid_from_range(night)
        middle_of_night = time_grid[len(time_grid) / 2]

        return night, middle_of_night

    def get_observing_windows(self):
        """
            Get observing windows for when the targets in the target list are above elv_lim
        :return:
        """
        if len(self.targets) == 0:
            print('no targets in the target list')
            return

        # print(self.targets)
        for target in self.targets:
            target.set_observability_windows(_elv_lim=self.elv_lim)

    def get_guide_stars(self, _guide_star_cat=u'I/337/gaia', _radius=30.0, _margin=30.0, _m_lim_gs=16.0,
                        _plot_field=False, _psf_fits=None):
        """
            Get astrometric guide stars for each observing window
        :return:
        """
        if len(self.targets) == 0:
            print('no targets in the target list')
            return

        if _plot_field and _psf_fits is not None:
            try:
                with fits.open(_psf_fits) as hdulist:
                    model_psf = hdulist[0].data
            except IOError:
                # couldn't load? use a simple Gaussian then
                model_psf = None
        else:
            model_psf = None

        for target in self.targets:
            target.set_guide_stars(_jpl_eph=self.inp['jpl_eph'], _guide_star_cat=_guide_star_cat,
                                   _station=self.observatory_pypride, _radius=_radius, _margin=_margin,
                                   _m_lim_gs=_m_lim_gs, _plot_field=_plot_field, _model_psf=model_psf)


class AsteroidDatabase(object):

    def __init__(self):
        self.database = None
        self.f_database = None
        self.path_local = None
        self.database_url = None

    def asteroid_database_update(self, n=1.0):
        """
            Fetch an asteroid database update

            JPL: http://ssd.jpl.nasa.gov/dat/ELEMENTS.NUMBR
            MPC: http://www.minorplanetcenter.net/iau/MPCORB/ + [MPCORB.DAT, PHA.txt, NEA.txt, ...]
        """
        # make sure not to try to run from superclass
        if (self.f_database is not None) and (self.database_url is not None):
            do_update = False
            if os.path.isfile(self.f_database):
                age = datetime.datetime.now() - \
                      datetime.datetime.utcfromtimestamp(os.path.getmtime(self.f_database))
                if age.days > n:
                    do_update = True
                    print('Asteroid database: {:s} is out of date, updating...'.format(self.f_database))
            else:
                do_update = True
                print('Database file: {:s} is missing, fetching...'.format(self.f_database))
            # if the file is older than n days:
            if do_update:
                try:
                    response = urllib2.urlopen(self.database_url)
                    with open(self.f_database, 'w') as f:
                        f.write(response.read())
                except Exception as err:
                    print(str(err))
                    traceback.print_exc()
                    pass

    @staticmethod
    def init_asteroid(_asteroid_db_entry):
        """
            Initialize Asteroid object from 'raw' db entry
        """

        AU_DE430 = 1.49597870700000000e+11  # m
        GSUN = 0.295912208285591100e-03 * AU_DE430 ** 3 / 86400.0 ** 2
        # convert AU to m:
        a = _asteroid_db_entry['a'] * AU_DE430
        e = _asteroid_db_entry['e']
        # convert deg to rad:
        i = _asteroid_db_entry['i'] * np.pi / 180.0
        w = _asteroid_db_entry['w'] * np.pi / 180.0
        Node = _asteroid_db_entry['Node'] * np.pi / 180.0
        M0 = _asteroid_db_entry['M0'] * np.pi / 180.0
        t0 = _asteroid_db_entry['epoch']
        H = _asteroid_db_entry['H']
        G = _asteroid_db_entry['G']

        return Asteroid(_asteroid_db_entry['name'], a, e, i, w, Node, M0, GSUN, t0, H, G)

    def load(self):
        """
            Load database into self.database
        :return:
        """
        raise NotImplementedError

    def get_one(self, _name):
        """
            Get one asteroid from database
        :return: single Asteroid object
        """
        raise NotImplementedError

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of Asteroid objects
        """
        raise NotImplementedError


class AsteroidDatabaseJPL(AsteroidDatabase):

    def __init__(self, _path_local='./', _f_database='ELEMENTS.NUMBR'):
        # initialize super class
        super(AsteroidDatabaseJPL, self).__init__()

        self.f_database = _f_database
        self.path_local = _path_local
        self.database_url = os.path.join('http://ssd.jpl.nasa.gov/dat/', _f_database)

        # update database if necessary:
        self.asteroid_database_update(n=1.0)

    def load(self):
        """
            Load database into self.database
        :return:
        """
        with open(os.path.join(self.path_local, self.f_database), 'r') as f:
            database = f.readlines()

        # remove header:
        start = [i for i, l in enumerate(database[:300]) if l[0:2] == '--']
        if len(start) > 0:
            database = database[start[0] + 1:]
        # remove empty lines:
        database = [l for l in database if len(l) > 5]

        dt = np.dtype([('name', '|S21'),
                       ('epoch', '<i8'), ('a', '<f8'),
                       ('e', '<f8'), ('i', '<f8'),
                       ('w', '<f8'), ('Node', '<f8'),
                       ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
        self.database = np.array([((l[0:25].strip(),) + tuple(map(float, l[25:].split()[:-2]))) for l in database[2:]],
                                 dtype=dt)

    def get_one(self, _name):
        """
            Get one asteroid from database
        :return: single Asteroid object
        """
        # remove underscores:
        if '_' in _name:
            _name = ' '.join(_name.split('_'))
        # remove brackets:
        for prntz in ('(', ')'):
            if prntz in _name:
                _name = _name.replace(prntz, '')

        if self.database is not None:
            try:
                asteroid_entry = self.database[self.database['name'] == _name]
                # initialize and return Asteroid object here:
                return self.init_asteroid(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None
        else:
            # database not read into memory? access it quickly then:
            try:
                # get the entry in question:
                with open(os.path.join(self.path_local, self.f_database)) as f:
                    lines = f.readlines()
                    entry = [line for line in lines if line.find(_name) != -1][0]

                dt = np.dtype([('name', '|S21'),
                               ('epoch', '<i8'), ('a', '<f8'),
                               ('e', '<f8'), ('i', '<f8'),
                               ('w', '<f8'), ('Node', '<f8'),
                               ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
                # this is for numbered asteroids:
                asteroid_entry = np.array([((entry[0:25].strip(),) + tuple(map(float, entry[25:].split()[:-2])))], dtype=dt)
                # initialize and return Asteroid object here:
                return self.init_asteroid(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of Asteroid objects
        """
        if self.database is None:
            self.load()

        asteroids = [self.init_asteroid(asteroid_entry) for asteroid_entry in self.database]
        return asteroids


class AsteroidDatabaseMPC(AsteroidDatabase):

    def __init__(self, _path_local='./', _f_database='PHA.txt'):
        super(AsteroidDatabaseMPC, self).__init__()

        self.f_database = _f_database
        self.path_local = _path_local
        self.database_url = os.path.join('http://www.minorplanetcenter.net/iau/MPCORB/', _f_database)

        # update database if necessary:
        self.asteroid_database_update(n=1.0)

    @staticmethod
    def unpack_epoch(epoch_str):
        """
            Unpack MPC epoch
        :param epoch_str:
        :return:
        """
        def l2num(l):
            try:
                num = int(l)
            except ValueError:
                num = ord(l) - 55
            return num

        centuries = {'I': '18', 'J': '19', 'K': '20'}

        epoch = '{:s}{:s}{:02d}{:02d}'.format(centuries[epoch_str[0]], epoch_str[1:3],
                                              l2num(epoch_str[3]), l2num(epoch_str[4]))

        # convert to mjd:
        epoch_datetime = datetime.datetime.strptime(epoch, '%Y%m%d')
        mjd = mjuliandate(epoch_datetime.year, epoch_datetime.month, epoch_datetime.day)

        return mjd

    def load(self):
        """
            Load database into self.database
        :return:
        """
        with open(os.path.join(self.path_local, self.f_database), 'r') as f:
            database = f.readlines()

        # remove header:
        start = [i for i, l in enumerate(database[:300]) if l[0:2] == '--']
        if len(start) > 0:
            database = database[start[0] + 1:]
        # remove empty lines:
        database = [l for l in database if len(l) > 5]

        dt = np.dtype([('designation', '|S21'), ('H', '<f8'), ('G', '<f8'),
                       ('epoch', '<f8'), ('M0', '<f8'), ('w', '<f8'),
                       ('Node', '<f8'), ('i', '<f8'), ('e', '<f8'),
                       ('n', '<f8'), ('a', '<f8'), ('U', '|S21'),
                       ('n_obs', '<f8'), ('n_opps', '<f8'), ('arc', '|S21'),
                       ('rms', '|S21'), ('name', '|S21'), ('last_obs', '|S21')
                       ])

        self.database = np.array([(str(entry[:7]).strip(),)
                                  + (float(entry[8:13]) if len(entry[8:13].strip()) > 0 else 20.0,)
                                  + (float(entry[14:19]) if len(entry[14:19].strip()) > 0 else 0.15,)
                                  + (self.unpack_epoch(str(entry[20:25])),)
                                  + (float(entry[26:35]),) + (float(entry[37:46]),)
                                  + (float(entry[48:57]),) + (float(entry[59:68]),) + (float(entry[70:79]),)
                                  + (float(entry[80:91]) if len(entry[80:91].strip()) > 0 else 0,)
                                  + (float(entry[92:103]) if len(entry[92:103].strip()) > 0 else 0,)
                                  + (str(entry[105:106]),)
                                  + (int(entry[117:122]) if len(entry[117:122].strip()) > 0 else 0,)
                                  + (int(entry[123:126]) if len(entry[123:126].strip()) > 0 else 0,)
                                  + (str(entry[127:136]).strip(),)
                                  # + (str(entry[137:141]),) + (str(entry[166:194]).strip().replace(' ', '_'),)
                                  + (str(entry[137:141]),) + (str(entry[166:194]).strip(),)
                                  + (str(entry[194:202]).strip(),)
                                 for entry in database], dtype=dt)

    def get_one(self, _name):
        """
            Get one asteroid from database
        :return: single Asteroid object
        """
        # remove underscores:
        if '_' in _name:
            _name = ' '.join(_name.split('_'))

        if self.database is not None:
            try:
                asteroid_entry = self.database[self.database['name'] == _name]
                # initialize and return Asteroid object here:
                if len(asteroid_entry) > 0:
                    return self.init_asteroid(asteroid_entry)
                else:
                    return None

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None
        else:
            # database not read into memory? access it quickly then:
            try:
                # get the entry in question:
                with open(os.path.join(self.path_local, self.f_database)) as f:
                    lines = f.readlines()
                    entry = [line for line in lines if line.find(_name) != -1][0]

                dt = np.dtype([('designation', '|S21'), ('H', '<f8'), ('G', '<f8'),
                               ('epoch', '<f8'), ('M0', '<f8'), ('w', '<f8'),
                               ('Node', '<f8'), ('i', '<f8'), ('e', '<f8'),
                               ('n', '<f8'), ('a', '<f8'), ('U', '|S21'),
                               ('n_obs', '<f8'), ('n_opps', '<f8'), ('arc', '|S21'),
                               ('rms', '|S21'), ('name', '|S21'), ('last_obs', '|S21')
                               ])
                asteroid_entry = np.array(
                                        [(str(entry[:7]).strip(),)
                                         + (float(entry[8:13]) if len(entry[8:13].strip()) > 0 else 20.0,)
                                         + (float(entry[14:19]) if len(entry[14:19].strip()) > 0 else 0.15,)
                                         + (self.unpack_epoch(str(entry[20:25])),)
                                         + (float(entry[26:35]),) + (float(entry[37:46]),)
                                         + (float(entry[48:57]),) + (float(entry[59:68]),) + (float(entry[70:79]),)
                                         + (float(entry[80:91]) if len(entry[80:91].strip()) > 0 else 0,)
                                         + (float(entry[92:103]) if len(entry[92:103].strip()) > 0 else 0,)
                                         + (str(entry[105:106]),)
                                         + (int(entry[117:122]) if len(entry[117:122].strip()) > 0 else 0,)
                                         + (int(entry[123:126]) if len(entry[123:126].strip()) > 0 else 0,)
                                         + (str(entry[127:136]).strip(),)
                                         # + (str(entry[137:141]),) + (str(entry[166:194]).strip().replace(' ', '_'),)
                                         + (str(entry[137:141]),) + (str(entry[166:194]).strip(),)
                                         + (str(entry[194:202]).strip(),)], dtype=dt)
                # initialize Asteroid object here:
                return self.init_asteroid(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of Asteroid objects
        """
        if self.database is None:
            self.load()

        asteroids = [self.init_asteroid(asteroid_entry) for asteroid_entry in self.database]
        return asteroids


class Asteroid(object):
    """
       Class to work with Keplerian orbits of Asteroids
    """

    def __init__(self, name, a, e, i, w, Node, M0, GM, t0, H=None, G=None):
        # a should be in metres, all the angles - in radians, t0 in mjd [days]
        # GM = G*(M_central_body + M_body) for 2 body problem
        # GM = G*M_central_body**3/ (M_central_body + M_body)**2
        #                   for restricted 3 body problem
        # GM = [m**2/kg/s**2]
        self.name = name
        self.a = float(a)
        self.e = float(e)
        self.i = float(i)
        self.w = float(w)
        self.Node = float(Node)
        self.M0 = float(M0)
        self.GM = float(GM)
        self.t0 = float(t0)
        self.H = float(H)
        self.G = float(G)

    def __str__(self):
        """
            Print it out nicely
        """
        return '<Keplerian object {:s}: a={:e} m, e={:f}, i={:f} rad, '. \
                   format(self.name, self.a, self.e, self.i) + \
               'w={:f} rad, Node={:f} rad, M0={:f} rad, '. \
                   format(self.w, self.Node, self.M0) + \
               't0={:f} (MJD), GM={:e} m**3/kg/s**2>'. \
                   format(self.t0, self.GM)

    @staticmethod
    @jit
    def kepler(e, M):
        """ Solve Kepler's equation

        :param e: eccentricity
        :param M: mean anomaly, rad
        :return:
        """
        E = deepcopy(M)
        tmp = 1

        while np.abs(E - tmp) > 1e-9:
            tmp = deepcopy(E)
            E += (M - E + e * np.sin(E)) / (1 - e * np.cos(E))

        return E

    @jit
    def to_cart(self, t):
        """
            Compute Cartesian state at epoch t with respect to the central body
            from Keplerian elements
            t -- epoch in mjd [decimal days]
        """
        # mean motion:
        n = np.sqrt(self.GM / self.a / self.a / self.a) * 86400.0  # [rad/day]
        # mean anomaly at t:
        M = n * (t - self.t0) + self.M0
        #        print(np.fmod(M, 2*np.pi))
        # solve Kepler equation, get eccentric anomaly:
        E = self.kepler(self.e, M)
        cosE = np.cos(E)
        sinE = np.sin(E)
        # get true anomaly and distance from focus:
        sinv = np.sqrt(1.0 - self.e ** 2) * sinE / (1.0 - self.e * cosE)
        cosv = (cosE - self.e) / (1.0 - self.e * cosE)
        r = self.a * (1.0 - self.e ** 2) / (1.0 + self.e * cosv)
        #        r = self.a*(1 - self.e*cosE)
        #
        sinw = np.sin(self.w)
        cosw = np.cos(self.w)
        sinu = sinw * cosv + cosw * sinv
        cosu = cosw * cosv - sinw * sinv
        # position
        cosNode = np.cos(self.Node)
        sinNode = np.sin(self.Node)
        cosi = np.cos(self.i)
        sini = np.sin(self.i)
        x = r * (cosu * cosNode - sinu * sinNode * cosi)
        y = r * (cosu * sinNode + sinu * cosNode * cosi)
        z = r * sinu * sini
        # velocity
        p = self.a * (1.0 - self.e ** 2)
        V_1 = np.sqrt(self.GM / p) * self.e * sinv
        V_2 = np.sqrt(self.GM / p) * (1.0 + self.e * cosv)
        vx = x * V_1 / r + (-sinu * cosNode - cosu * sinNode * cosi) * V_2
        vy = y * V_1 / r + (-sinu * sinNode + cosu * cosNode * cosi) * V_2
        vz = z * V_1 / r + cosu * sini * V_2

        state = np.array([x, y, z, vx, vy, vz])
        state = np.reshape(np.asarray(state), (3, 2), 'F')

        return state

    @staticmethod
    @jit
    def ecliptic_to_equatorial(state):
        """
            epsilon at J2000 = 23.439279444444445 - from DE430
        """
        # transformation matrix
        eps = 23.439279444444444 * np.pi / 180.0
        # eps = 23.43929111*np.pi/180.0
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(eps), -np.sin(eps)],
                      [0.0, np.sin(eps), np.cos(eps)]])
        r = np.dot(R, state[:, 0])
        v = np.dot(R, state[:, 1])
        state = np.hstack((r, v))
        state = np.reshape(np.asarray(state), (3, 2), 'F')

        return state

    @staticmethod
    def PNmatrix(t, inp):
        """
            Compute (geocentric) IAU2000 precession/nutation matrix for epoch
            t -- astropy.time.Time object
        """
        from pypride.vintlib import taitime, eop_iers, t_eph, ter2cel, load_cats
        # precess to date
        ''' set dates: '''

        tstamp = t.datetime
        mjd = np.floor(t.mjd)
        UTC = (tstamp.hour + tstamp.minute / 60.0 + tstamp.second / 3600.0) / 24.0
        JD = mjd + 2400000.5

        ''' compute tai & tt '''
        TAI, TT = taitime(mjd, UTC)

        ''' load cats '''
        _, _, eops = load_cats(inp, 'DUMMY', 'S', ['GEOCENTR'], tstamp)

        ''' interpolate eops to tstamp '''
        UT1, eop_int = eop_iers(mjd, UTC, eops)

        ''' compute coordinate time fraction of CT day at GC '''
        CT, dTAIdCT = t_eph(JD, UT1, TT, 0.0, 0.0, 0.0)

        ''' rotation matrix IERS '''
        r2000 = ter2cel(tstamp, eop_int, dTAIdCT, 'iau2000')
        #    print(r2000[:,:,0])

        return r2000

    # @jit
    def raDecVmag(self, mjd, jpl_eph, epoch='J2000', station=None, output_Vmag=False):
        """ Calculate ra/dec's from equatorial state
            Then compute asteroid's expected visual magnitude

        :param mjd: MJD epoch in decimal days
        :param jpl_eph: target's heliocentric equatorial
        :param epoch: RA/Dec epoch. 'J2000', 'Date' or float (like 2015.0)
        :param station: None or pypride station object
        :param output_Vmag: return Vmag?

        :return: SkyCoord(ra,dec), ra/dec rates, Vmag
        """

        # J2000 ra/dec's:
        jd = mjd + 2400000.5
        # Earth:
        rrd = pleph(jd, 3, 12, jpl_eph)
        earth = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
        # Sun:
        rrd = pleph(jd, 11, 12, jpl_eph)
        sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
        # target state:
        state = self.ecliptic_to_equatorial(self.to_cart(mjd))

        # quick and dirty LT computation (but accurate enough for pointing, I hope)
        # LT-correction:
        C = 299792458.0
        if station is None:
            lt = np.linalg.norm(earth[:, 0] - (sun[:, 0] + state[:, 0])) / C
        else:
            lt = np.linalg.norm((earth[:, 0] + station.r_GCRS) - (sun[:, 0] + state[:, 0])) / C
        # print(lt)

        # recompute:
        # Sun:
        rrd = pleph(jd - lt / 86400.0, 11, 12, jpl_eph)
        sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
        # target state:
        state = self.ecliptic_to_equatorial(self.to_cart(mjd - lt / 86400.0))

        if station is None:
            lt = np.linalg.norm(earth[:, 0] - (sun[:, 0] + state[:, 0])) / C
        else:
            lt = np.linalg.norm((earth[:, 0] + station.r_GCRS) - (sun[:, 0] + state[:, 0])) / C
        # print(lt)

        # recompute again:
        # Sun:
        rrd = pleph(jd - lt / 86400.0, 11, 12, jpl_eph)
        sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
        # target state:
        state = self.ecliptic_to_equatorial(self.to_cart(mjd - lt / 86400.0))

        # geocentric/topocentric RA/Dec
        if station is None:
            r = (sun[:, 0] + state[:, 0]) - earth[:, 0]
        else:
            r = (sun[:, 0] + state[:, 0]) - (earth[:, 0] + station.r_GCRS)
        # RA/Dec J2000:
        ra = np.arctan2(r[1], r[0])  # right ascension
        dec = np.arctan(r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2))  # declination
        if ra < 0:
            ra += 2.0 * np.pi

        # go for time derivatives:
        if station is None:
            v = (sun[:, 1] + state[:, 1]) - earth[:, 1]
        else:
            v = (sun[:, 1] + state[:, 1]) - (earth[:, 1] + station.v_GCRS)
        # in rad/s:
        ra_dot = (v[1] / r[0] - r[1] * v[0] / r[0] ** 2) / (1 + (r[1] / r[0]) ** 2)
        dec_dot = (v[2] / np.sqrt(r[0] ** 2 + r[1] ** 2) -
                   r[2] * (r[0] * v[0] + r[1] * v[1]) / (r[0] ** 2 + r[1] ** 2) ** 1.5) / \
                  (1 + (r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2)) ** 2)
        # convert to arcsec/h:
        # ra_dot = ra_dot * 180.0 / np.pi * 3600.0 * 3600.0
        # dec_dot = dec_dot * 180.0 / np.pi * 3600.0 * 3600.0
        # convert to arcsec/s:
        ra_dot = ra_dot * 180.0 / np.pi * 3600.0
        dec_dot = dec_dot * 180.0 / np.pi * 3600.0
        # ra_dot*cos(dec), dec_dot:
        #        print(ra_dot * np.cos(dec), dec_dot)

        # RA/Dec to date:
        if epoch != 'J2000':
            print(ra, dec)
            xyz2000 = sph2cart(np.array([1.0, dec, ra]))
            if epoch != 'Date' and isinstance(epoch, float):
                jd = Time(epoch, format='jyear').jd
            rDate = iau_PNM00A(jd, 0.0)
            xyzDate = np.dot(rDate, xyz2000)
            dec, ra = cart2sph(xyzDate)[1:]
            if ra < 0:
                ra += 2.0 * np.pi
            print(ra, dec)

        ''' go for Vmag based on H-G model '''
        if self.H is None or self.G is None:
            print('Can\'t compute Vmag - no H-G model data provided.')
            Vmag = None
        elif not output_Vmag:
            Vmag = None
        else:
            # phase angle:
            EA = r
            SA = state[:, 0]
            EA_norm = np.linalg.norm(EA)
            SA_norm = np.linalg.norm(SA)
            alpha = np.arccos(np.dot(EA, SA) / (EA_norm * SA_norm))
            #            print(alpha)
            #            phi1 = np.exp(-3.33*np.sqrt(np.tan(alpha))**0.63)
            #            phi2 = np.exp(-1.87*np.sqrt(np.tan(alpha))**1.22)

            W = np.exp(-90.56 * np.tan(alpha / 2.0) ** 2)
            phi1s = 1 - 0.986 * np.sin(alpha) / \
                        (0.119 + 1.341 * np.sin(alpha) - 0.754 * np.sin(alpha) ** 2)
            phi1l = np.exp(-3.332 * (np.tan(alpha / 2.0)) ** 0.631)
            phi1 = W * phi1s + (1.0 - W) * phi1l

            phi2s = 1 - 0.238 * np.sin(alpha) / \
                        (0.119 + 1.341 * np.sin(alpha) - 0.754 * np.sin(alpha) ** 2)
            phi2l = np.exp(-1.862 * (np.tan(alpha / 2.0)) ** 1.218)
            phi2 = W * phi2s + (1.0 - W) * phi2l

            AU_DE430 = 1.49597870700000000e+11  # m

            Vmag = self.H - 2.5 * np.log10((1.0 - self.G) * phi1 + self.G * phi2) + \
                   5.0 * np.log10(EA_norm * SA_norm / AU_DE430 ** 2)

        # returning SkyCoord is handy, but very expensive
        # return (SkyCoord(ra=ra, dec=dec, unit=(u.rad, u.rad), frame='icrs'),
        #         (ra_dot, dec_dot), Vmag)
        return [ra, dec], [ra_dot, dec_dot], Vmag


if __name__ == '__main__':
    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(abs_path, 'config.ini'))

    f_inp = config.get('Path', 'pypride_inp')

    # observatory and time zone
    observatory = config.get('Observatory', 'observatory')
    timezone = config.get('Observatory', 'timezone')

    # observability settings:
    # nighttime between twilights: astronomical (< -18 deg), civil (< -6 deg), nautical (< -12 deg)
    twilight = config.get('Asteroids', 'twilight')
    # fraction of night when observable given constraints:
    fraction = float(config.get('Asteroids', 'fraction'))
    # magnitude limit:
    m_lim = float(config.get('Asteroids', 'm_lim'))
    # elevation cut-off [deg]:
    elv_lim = float(config.get('Asteroids', 'elv_lim'))

    # guide star settings:
    guide_star_cat = config.get('Vizier', 'guide_star_cat')
    radius = float(config.get('Vizier', 'radius'))
    margin = float(config.get('Vizier', 'margin'))
    m_lim_gs = float(config.get('Vizier', 'm_lim_gs'))
    plot_field = eval(config.get('Vizier', 'plot_field'))
    psf_fits = config.get('Vizier', 'psf_fits')

    run_tests = False
    if run_tests:
        print('running tests...')
        # run tests:
        # asteroid name must be MPC-friendly
        asteroid_name = '(3200)_Phaethon'

        print('testing the JPL database')
        jpl_db = AsteroidDatabaseJPL()
        jpl_db.load()
        asteroid_jpl = jpl_db.get_one(_name=asteroid_name)
        print(asteroid_jpl)

        print('testing the MPC database')
        mpc_db = AsteroidDatabaseMPC()
        mpc_db.load()
        asteroid_mpc = mpc_db.get_one(_name=asteroid_name)
        print(asteroid_mpc)
        if True is False:
            asteroids_mpc = mpc_db.get_all()
            for asteroid in asteroids_mpc:
                print(asteroid)

    time_str = '20161022_042047.042560'
    start_time = Time(str(datetime.datetime.strptime(time_str, '%Y%m%d_%H%M%S.%f')), format='iso', scale='utc')

    ''' Target list '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone(timezone))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

    tl = TargetListAsteroids(f_inp, database_source='mpc', database_file='PHA.txt',
                             _observatory=observatory, _m_lim=m_lim, _elv_lim=elv_lim, date=today)
    # get all bright targets given m_lim
    mask = None
    tl.target_list_all(today, mask, parallel=True)
    # get observable given elv_lim
    tl.target_list_observable(today, twilight=twilight, fraction=fraction)
    # get observing windows
    tl.get_observing_windows()
    # get guide stars
    tl.get_guide_stars(_guide_star_cat=guide_star_cat, _radius=radius, _margin=margin, _m_lim_gs=m_lim_gs,
                       _plot_field=plot_field, _psf_fits=psf_fits)

    for tr in tl.targets:
        print(tr)
