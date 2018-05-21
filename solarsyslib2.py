from __future__ import print_function
import numpy as np
import os
import datetime
import urllib2
import ConfigParser
import requests
import argparse
import json
import inspect
import lxml.html as lh
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dicttoxml import dicttoxml
from collections import OrderedDict
from pypride.vintflib import pleph, lagint
from pypride.classes import inp_set, constants
from pypride.vintlib import eop_update, internet_on, cel2ter00, mjuliandate, taitime, eop_iers, t_eph, nsec
from pypride.vintlib import sph2cart, cart2sph, iau_PNM00A
from pypride.vintlib import factorise, aber_source, R_123, iau_RX, iau_RY, iau_RZ
from pypride.vintlib import memoize
from astroplan import Observer as Observer_astroplan

from astropy.utils.data import clear_download_cache
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.units import Quantity
from astropy import table
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.time import Time
from time import time as _time
import pytz
from numba import jit
import math
from copy import deepcopy
import traceback
from astroquery.query import suspend_cache
from astroquery.gaia import Gaia
import multiprocessing
import gc
# from distributed import Client, LocalCluster

from penquins import Kowalski

import warnings

import matplotlib

# to display stuff internally:
# matplotlib.use('Qt5Agg')
# to run externally without X-server:
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import aplpy

# import image_registration

warnings.filterwarnings("ignore")

# use dask.distributed to run stuff in parallel

# initialize a local 'cluster':
# cluster = LocalCluster()
# client = Client(cluster)
# client = Client('127.0.0.1:8786')

# initialize Gaia object
with suspend_cache(Gaia):
    gaia = Gaia
    # gaia.ROW_LIMIT = -1
    # gaia.TIMEOUT = 30

# init kowalski:
# kowalski = Kowalski(username='duev', password='PWD')


def is_planet_or_moon(name):
    """

    :param name: body name
    :return:
    """
    planets = ('mercury', 'venus', 'earth', 'mars', 'jupiter',
               'saturn', 'uranus', 'neptune')
    moons = ('moon',
             'deimos', 'phobos',
             'europa', 'io', 'ganymede', 'callisto',
             'titan', 'enceladus', 'dione', 'hyperion', 'iapetus', 'mimas', 'rhea', 'tethys',
             'miranda', 'ariel', 'umbriel', 'oberon', 'titania')

    if name.lower() in planets or 'pluto' in name.lower() or name.lower() in moons:
        return True
    else:
        return False


def is_multiple_asteroid(name):
    """

    :param name:
    :param _f_base:
    :return:
    """
    try:
        # as of Nov 15, 2016
        binary_list = 'http://www.johnstonsarchive.net/astro/asteroidmoonslist2.html'

        response = urllib2.urlopen(binary_list)
        response_html = response.read()

        doc = lh.document_fromstring(response_html)
        a = doc.xpath('./body/center/table/tr/td/ul/li/a')
        binaries = ['_'.join(aa.text_content().split(',')[0].split('and')[0]
                             .strip().split()).replace('(', '').replace(')', '') for aa in a]

    except Exception as err:
        print(str(err))
        print('could not get the binary asteroid list.')
        return False

    if name in binaries:
        return True
    else:
        return False


def GTRS_to_GCRS(date_utc, eop_int, dTAIdCT=1.0, mode='der1'):
    """
    Compute celestial-to-terrestrial matrix for a given UTC date/time

    mode='der' - calculate derivatives
    mode='der1' - calculate only first derivative
    mode='noDer' - do not calculate derivatives
    """
    if mode == 'der':
        r2000 = np.zeros((3, 3, 3))
    elif mode == 'der1':
        r2000 = np.zeros((3, 3, 2))
    elif mode == 'noDer':
        r2000 = np.zeros((3, 3))
    else:
        r2000 = np.zeros((3, 3))

    sec = datetime.timedelta(seconds=1)
    # following IAU 2000A:
    cel2ter = cel2ter00(date_utc, eop_int)
    # numerically calculate derivatives:
    if mode == 'der':
        cel2ter_m2 = cel2ter00(date_utc - 2 * sec, eop_int)  # -2s TAI
        cel2ter_m1 = cel2ter00(date_utc - sec, eop_int)  # -1s TAI
        cel2ter_p1 = cel2ter00(date_utc + sec, eop_int)  # +1s TAI
        cel2ter_p2 = cel2ter00(date_utc + 2 * sec, eop_int)  # +2s TAI
    elif mode == 'der1':
        cel2ter_m1 = cel2ter00(date_utc - sec, eop_int)  # -1s TAI
        cel2ter_p1 = cel2ter00(date_utc + sec, eop_int)  # +1s TAI
    else:
        cel2ter_m1 = np.zeros((3, 3))
        cel2ter_p1 = np.zeros((3, 3))
        cel2ter_m2 = np.zeros((3, 3))
        cel2ter_p2 = np.zeros((3, 3))

    if mode == 'der':
        # transpose it to get terrestrial-to-celestial matrix and its CT derivative
        r2000[:, :, 0] = cel2ter.T
        # (dr/dTAI)*(dTAI/dCT):
        r2000[:, :, 1] = dTAIdCT * (cel2ter_p1.T - cel2ter_m1.T) / 2.0
        # (dv/dTAI)*(dTAI/dCT):
        r2000[:, :, 2] = dTAIdCT * (cel2ter_p2.T -
                                    2.0 * cel2ter.T + cel2ter_m2.T) / 4.0
    elif mode == 'der1':
        # transpose it to get terrestrial-to-celestial matrix and its CT derivative
        r2000[:, :, 0] = cel2ter.T
        # (dr/dTAI)*(dTAI/dCT):
        r2000[:, :, 1] = dTAIdCT * (cel2ter_p1.T - cel2ter_m1.T) / 2.0
    elif mode == 'noDer':
        r2000 = cel2ter.T
    else:
        raise Exception('unrecognised mode')

    return r2000


class Site(object):
    """
        Quick and dirty!
    """

    def __init__(self, r_GTRS):
        if not isinstance(r_GTRS, (np.ndarray, np.generic)):
            r_GTRS = np.array(r_GTRS)
        self.r_GTRS = r_GTRS
        self.r_GCRS = None
        self.v_GCRS = None
        self.r_BCRS = None
        self.v_BCRS = None

        # IERS 2010
        AE = 6378136.3  # m
        F = 298.25642

        # Compute the site spherical radius
        self.sph_rad = np.sqrt(sum(x * x for x in self.r_GTRS))
        # Compute geocentric latitude
        self.lat_gcen = np.arcsin(self.r_GTRS[2] / self.sph_rad)
        # Compute geocentric longitude
        self.lon_gcen = np.arctan2(self.r_GTRS[1], self.r_GTRS[0])
        if self.lon_gcen < 0.0:
            self.lon_gcen += 2.0 * np.pi
        # Compute the stations distance from the Earth spin axis and
        # from the equatorial plane (in KM)
        req = np.sqrt(sum(x * x for x in self.r_GTRS[:-1]))
        self.u = req * 1e-3
        self.v = self.r_GTRS[2] * 1e-3
        # Compute geodetic latitude and height.
        # The geodetic longitude is equal to the geocentric longitude
        self.lat_geod, self.h_geod = self.geoid(req, self.r_GTRS[2], AE, F)
        # Compute the local VEN-to-crust-fixed rotation matrices by rotating
        # about the geodetic latitude and the longitude.
        # w - rotation matrix by an angle lat_geod around the y axis
        w = R_123(2, self.lat_geod)
        # v - rotation matrix by an angle -lon_gcen around the z axis
        v = R_123(3, -self.lon_gcen)
        # product of the two matrices:
        self.vw = np.dot(v, w)

    @staticmethod
    def geoid(r, z, a, fr):
        """
        Transform Cartesian to geodetic coordinates
        based on the exact solution (Borkowski,1989)

               Input variables :
                     r, z = equatorial [m] and polar [m] components
               Output variables:
                     fi, h = geodetic coord's (latitude [rad], height [m])

        IERS ellipsoid: semimajor axis (a) and inverse flattening (fr)
        """

        if z >= 0.0:
            b = abs(a - a / fr)
        else:
            b = -abs(a - a / fr)
        E = ((z + b) * b / a - a) / r
        F = ((z - b) * b / a + a) / r
        # Find solution to: t**4 + 2*E*t**3 + 2*F*t - 1 = 0
        P = (E * F + 1.0) * 4.0 / 3.0
        Q = (E * E - F * F) * 2.0
        D = P * P * P + Q * Q
        if D >= 0.0:
            s = np.sqrt(D) + Q
            if s >= 0:
                s = abs(np.exp(np.log(abs(s)) / 3.0))
            else:
                s = -abs(np.exp(np.log(abs(s)) / 3.0))
            v = P / s - s
            # Improve the accuracy of numeric values of v
            v = -(Q + Q + v * v * v) / (3.0 * P)
        else:
            v = 2.0 * np.sqrt(-P) * np.cos(np.arccos(Q / P / np.sqrt(-P)) / 3.0)

        G = 0.5 * (E + np.sqrt(E * E + v))
        t = np.sqrt(G * G + (F - v * G) / (G + G - E)) - G
        fi = np.arctan2((1.0 - t * t) * a, (2.0 * b * t))
        h = np.abs((r - a * t) * np.cos(fi) + (z - b) * np.sin(fi))

        return fi, h

    def GTRS_to_GCRS(self, r2000):
        self.r_GCRS = np.dot(r2000[:, :, 0], self.r_GTRS)
        self.v_GCRS = np.dot(r2000[:, :, 1], self.r_GTRS)


def geodetic2Ecef(lat, lon, h, a=None, b=None):
    """
    Converts geodetic to Earth Centered, Earth Fixed coordinates.

    Parameters h, a, and b must be given in the same unit.
    The values of the return tuple then also have this unit.

    :param lat: latitude(s) in radians
    :param lon: longitude(s) in radians
    :param h: height(s)
    :param a: equatorial axis of the ellipsoid of revolution
    :param b: polar axis of the ellipsoid of revolution
    :rtype: tuple (x,y,z)
    """
    if a is None and b is None:
        WGS84_a = 6378137.0  # meters
        """the equatorial radius in meters of the WGS84 ellipsoid in meters"""
        WGS84_f = 1 / 298.257223563
        """the flattening of the WGS84 ellipsoid, 1/298.257223563"""

        wgs84A = WGS84_a / 1000
        wgs84B = wgs84A * (1 - WGS84_f)

        a, b = wgs84A, wgs84B

    lat, lon, h = np.asarray(lat), np.asarray(lon), np.asarray(h)
    e2 = (a * a - b * b) / (a * a)  # first eccentricity squared
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    latCos = np.cos(lat)
    nh = n + h
    x = nh * latCos * np.cos(lon)
    y = nh * latCos * np.sin(lon)
    z = (n * (1 - e2) + h) * np.sin(lat)

    return x, y, z


def rotation(_vec, _radec_start, _angle_stop):
    """
        Align vector _vec with a coordinate system defined by a great circle segment on
        the celestial sphere
        Transform Cartesian vector _vec to a coordinate system, whose X axis is aligned with
        _radec_start (start point of the segment) and the endpoint of the segment lies in the XY plane
         so that the coordinates of the stop point are (cos(theta), sin(theta), 0), where
         theta is the angle between radec_start and radec_stop
    :param _vec:
    :param _radec_start:
    :param _angle_stop:
    :return:
    """
    return iau_RX(_angle_stop, iau_RY(-_radec_start[1], iau_RZ(_radec_start[0], _vec)))


def rotate_radec(_radec, _radec_start, _angle_stop):
    """
        Rotate _radec to the system described in the definition of rotation()
    :param _radec: [RA, Dec] in rad
    :param _radec_start: [RA, Dec] in rad
    :param _angle_stop: rad
    :return: [lon, lat] in rad (corresponding to RA and Dec)
    """
    rdecra = np.hstack([1.0, _radec[::-1]])
    cart = sph2cart(rdecra)
    return cart2sph(rotation(cart, _radec_start, _angle_stop))[:-3:-1]


def derotation(_vec, _radec_start, _angle_stop):
    """
        Transformation, inverse to rotation()
    :param _vec:
    :param _radec_start:
    :param _angle_stop:
    :return:
    """
    return iau_RZ(-_radec_start[0], iau_RY(_radec_start[1], iau_RX(-_angle_stop, _vec)))


def derotate_lonlat(_lonlat, _radec_start, _angle_stop):
    """
        Rotate _lonlat to the system described in the definition of derotation()
    :param _lonlat: [lon, lat] in rad
    :param _radec_start: [RA, Dec] in rad
    :param _angle_stop: rad
    :return: [lon, lat] in rad (corresponding to RA and Dec)
    """
    rlatlon = np.hstack([1.0, _lonlat[::-1]])
    cart = sph2cart(rlatlon)
    _radec = cart2sph(derotation(cart, _radec_start, _angle_stop))[:-3:-1]
    if _radec[0] < 0:
        _radec[0] += 2.0 * np.pi
    return _radec


def split_great_circle_segment(_beg, _end, _separation):
    """
        Iteratively split great circle segment until separation between consecutive points is <=_separation
    :param _beg:
    :param _end:
    :param _separation: in arcseconds
    :return:
    """
    result = []

    def split_gc(_beg, _end, _separation):

        # print(_beg.separation(_end))
        if _beg.separation(_end).deg >= _separation / 3600.0:
            _middle, _ = great_circle_segment_midpoint(_beg, _end)
            # left half
            split_gc(_beg, _middle, _separation)
            # right half
            split_gc(_middle, _end, _separation)
        else:
            if _beg not in result:
                result.append(_beg)
            if _end not in result:
                result.append(_end)

    split_gc(_beg, _end, _separation)

    return result


def great_circle_segment_midpoint(_beg, _end):
    """
        'Split' great circle segment in halves
    :param _beg: astropy SkyCoord instance, coordinates of 'start' point
    :param _end: astropy SkyCoord instance, coordinates of 'end' point
    :return:
    """
    # first convert RA/Dec's on a unit sphere to Cartesian coordinates:
    radec_beg = [_beg.ra.rad, _beg.dec.rad]
    rdecra_beg = np.hstack([1.0, radec_beg[::-1]])
    cart_beg = sph2cart(rdecra_beg)

    radec_end = [_end.ra.rad, _end.dec.rad]
    rdecra_end = np.hstack([1.0, radec_end[::-1]])
    cart_end = sph2cart(rdecra_end)

    # compute midpoint of the _shorter_ segment of GC passing through _beg and _end
    # negate the result to get the same for the _longer_ segment
    lamb = 1 + np.dot(cart_beg, cart_end) / 1.0 ** 2
    middle_cart = (cart_beg + cart_end) / np.sqrt(2.0 * lamb)
    rdecra_middle = cart2sph(middle_cart)
    radec_middle = rdecra_middle[:-3:-1]
    # print(radec_middle)

    if radec_middle[0] < 0:
        radec_middle[0] += 2.0 * np.pi

    middle = SkyCoord(ra=radec_middle[0], dec=radec_middle[1], unit=[u.rad, u.rad], frame='icrs')

    return middle, radec_middle


@jit
def great_circle_distance(phi1, lambda1, phi2, lambda2):
    # input: dec1, ra1, dec2, ra2 [rad]
    # this is orders of magnitude faster than astropy.coordinates.Skycoord.separation
    delta_lambda = np.abs(lambda2 - lambda1)
    return np.arctan2(np.sqrt((np.cos(phi2) * np.sin(delta_lambda)) ** 2
                              + (
                              np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)) ** 2),
                      np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda))


@jit
def find_min(fun):
    """
      Find function minimum using a cubic fit
    :param fun:
    :return:
    """

    # mx = np.max(fun)
    # jmax = np.argmax(fun)
    mx = np.min(fun)
    jmax = np.argmin(fun)

    # first or last element?
    if jmax == 0 or jmax == len(fun) - 1:
        return jmax, mx, jmax, mx

    # if not - make a cubic fit
    a2 = 0.5 * (fun[jmax - 1] + fun[jmax + 1] - 2.0 * fun[jmax])
    a1 = 0.5 * (fun[jmax + 1] - fun[jmax - 1])
    djx = -a1 / (2.0 * a2)
    xmax = jmax + djx

    yfit = np.polyfit([-1, 0, 1], fun[jmax - 1: jmax + 2], 2)
    ymax = np.polyval(yfit, djx)
    # dydx = np.polyval(np.polyder(yfit), djx)  # must be 0
    # print(mx, jmax, xmax, ymax, dydx)

    return jmax, mx, xmax, ymax


@jit
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


@jit
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


# @jit
def dms(rad):
    d, m = divmod(abs(rad), np.pi / 180)
    m, s = divmod(m, np.pi / 180 / 60)
    s /= np.pi / 180 / 3600
    if rad >= 0:
        return [d, m, s]
    else:
        return [-d, -m, -s]


# @jit
def hms(rad):
    if rad < 0:
        rad += np.pi
    h, m = divmod(rad, np.pi / 12)
    m, s = divmod(m, np.pi / 12 / 60)
    s /= np.pi / 12 / 3600
    return [h, m, s]


def days_to_hmsm(days):
    """
    Convert fractional days to hours, minutes, seconds, and microseconds.
    Precision beyond microseconds is rounded to the nearest microsecond.

    Parameters
    ----------
    days : float
        A fractional number of days. Must be less than 1.

    Returns
    -------
    hour : int
        Hour number.

    min : int
        Minute number.

    sec : int
        Second number.

    micro : int
        Microsecond number.

    Raises
    ------
    ValueError
        If `days` is >= 1.

    Examples
    --------
    >>> days_to_hmsm(0.1)
    (2, 24, 0, 0)

    """
    hours = days * 24.
    hours, hour = math.modf(hours)

    mins = hours * 60.
    mins, min = math.modf(mins)

    secs = mins * 60.
    secs, sec = math.modf(secs)

    micro = round(secs * 1.e6)

    return int(hour), int(min), int(sec), int(micro)


def jd_to_date(jd):
    """
    Convert Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.

    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.

    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)

    """
    jd += 0.5

    F, I = math.modf(jd)
    I = int(I)

    A = math.trunc((I - 1867216.25) / 36524.25)

    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + F - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day


def jd_to_datetime(jd):
    """
    Convert a Julian Day to an `jdutil.datetime` object.

    Parameters
    ----------
    jd : float
        Julian day.

    Returns
    -------
    dt : `jdutil.datetime` object
        `jdutil.datetime` equivalent of Julian day.

    Examples
    --------
    >>> jd_to_datetime(2446113.75)
    datetime(1985, 2, 17, 6, 0)

    """
    year, month, day = jd_to_date(jd)

    frac_days, day = math.modf(day)
    day = int(day)

    hour, min, sec, micro = days_to_hmsm(frac_days)

    return datetime.datetime(year, month, day, hour, min, sec, micro)


def _target_is_vector(target):
    if hasattr(target, '__iter__'):
        return True
    else:
        return False


def target_list_helper(args):
    """
        Helper function to run target list computation in parallel
    :param args:
    :return:
    """
    try:
        targlist, asteroid, mjds, _epoch, _output_Vmag, night_grid, _twilight, _fraction = args

        # target can move fast, so compute radec/dots on a grid for the whole night
        radecs, radec_dots, Vmags = [], [], []
        for mjd in mjds:
            radec, radec_dot, Vmag = targlist.get_obs_params(asteroid, mjd, _epoch, _output_Vmag)
            radecs.append(radec)
            radec_dots.append(radec_dot)
            Vmags.append(Vmag)
        radecs = np.array(radecs)
        radec_dots = np.array(radec_dots)
        Vmags = np.array(Vmags)

        # skip if too dim
        if np.min(Vmags) <= targlist.m_lim:
            observable, meridian_transit, azels = targlist.observability(night_grid, radecs, Vmags,
                                                                         twilight=_twilight, fraction=_fraction)
            # print(asteroid['name'], observable, meridian_transit)
            return [MinorBodyDatabase.init_minor_body(asteroid), observable, meridian_transit,
                    radecs, radec_dots, azels, Vmags]
        else:
            return None
    except Exception as e:
        print(e)
        return None


# overload target meridian transit
class Observer(Observer_astroplan):
    # def __init__(self):
    #     # initialize super class
    #     super(Observer_astroplan, self).__init__()

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


def load_eops(_cat_eop, _date):
    # get the relevant eop entries from the catalog:
    with open(_cat_eop, 'r') as fc:
        fc_lines = fc.readlines()
    mjd_start = mjuliandate(_date.year, _date.month, _date.day)
    eops = np.zeros((7, 7))  # +/- 3 days
    for jj, line in enumerate(fc_lines):
        if line[0] not in (' ', '*', '#'):
            entry = [float(x) for x in line.split()]
            if len(entry) > 0 and entry[3] == np.floor(mjd_start) - 3:
                for kk in range(7):
                    eops[kk, 0] = entry[3]  # mjd
                    eops[kk, 1] = entry[6]  # UT1-UTC
                    eops[kk, 2] = entry[6] - nsec(entry[3])  # UT1 - TAI
                    eops[kk, 3] = entry[4]  # x_p
                    eops[kk, 4] = entry[5]  # y_p
                    eops[kk, 5] = entry[8]  # dX
                    eops[kk, 6] = entry[9]  # dY
                    entry = [float(x) for x in fc_lines[jj + kk + 1].split()]
                break  # exit loop

    return eops


class Target(object):
    """
        Store data for a single object at epoch
    """

    def __init__(self, _object):
        self.object = _object
        # position at the middle of the night:
        self.mean_position = {'epoch': None, 'radec': None, 'radec_dot': None, 'mag': None}
        # this is to keep [time_grid ra dec ra_dot dec_dot az el Vmag]:
        self.ephemeris = None
        self.meridian_crossing = None
        self.is_observable = None
        self.observability_windows = None
        self.guide_stars = None

    def __str__(self):
        """
            Print it out nicely
        """
        out_str = '<<Target object: {:s} '.format(str(self.object))
        if self.mean_position['epoch'] is not None:
            out_str += '\n mean epoch: {:s}'.format(str(self.mean_position['epoch']))
        if self.mean_position['mag'] is not None:
            out_str += '\n magnitude: {:.3f}'.format(self.mean_position['mag'])
        if self.mean_position['radec'] is not None:
            out_str += '\n ra: {:.10f} rad, dec: {:.10f} rad'.format(*self.mean_position['radec'])
        if self.mean_position['radec_dot'] is not None:
            out_str += '\n ra_dot: {:.3f} arcsec/sec, dec_dot: {:.3f} arcsec/sec'.format(
                *self.mean_position['radec_dot'])
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
        return out_str + '>>'

    def to_dict(self):
        """
            to be jsonified
        :return:
        """
        # raise NotImplementedError
        if self.ephemeris is None:
            raise Exception('compute ephemeris first, aborting.')

        mean_epoch = np.mean([ei.jyear for ei in self.ephemeris[:, 0]])
        mean_ra = np.mean(self.ephemeris[:, 1])
        mean_dec = np.mean(self.ephemeris[:, 2])
        mean_ra_dot = np.mean(self.ephemeris[:, 3])
        mean_dec_dot = np.mean(self.ephemeris[:, 4])
        mean_mag = np.mean(self.ephemeris[:, -1])

        # mean_epoch_str = self.epoch.datetime.strftime('%Y%m%d_%H%M%S.%f')
        mean_epoch_str = '{:.11f}'.format(mean_epoch)
        radec = [hms(mean_ra), dms(mean_dec)]
        ra_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(*radec[0])
        if radec[1][0] >= 0:
            dec_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))
        else:
            dec_str = '{:03.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))

        out_dict = OrderedDict([['name', self.object.name],
                                ['mean_magnitude', '{:.3f}'.format(mean_mag)],
                                ['mean_epoch', mean_epoch_str],
                                ['mean_radec', [ra_str, dec_str]],
                                ['mean_radec_dot', ['{:.5f}'.format(mean_ra_dot),
                                                    '{:.5f}'.format(mean_dec_dot)]],
                                ['meridian_crossing', str(self.meridian_crossing).lower()],
                                ['is_observable', str(self.is_observable).lower()],
                                ['guide_stars', []]
                                ])
        # save the date:
        out_dict['comment'] = 'modified_{:s}_UTC'.format(str(datetime.datetime.utcnow()))
        if self.guide_stars is not None:
            for star in self.guide_stars[::-1]:
                radec = [hms(star[1][0] * np.pi / 180), dms(star[1][1] * np.pi / 180)]
                ra_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(*radec[0])
                if radec[1][0] >= 0:
                    dec_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))
                else:
                    dec_str = '{:03.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))

                out_dict['guide_stars'].append(OrderedDict([['id', star[0]],
                                                            ['radec', [ra_str, dec_str]],
                                                            ['magnitude', '{:.3f}'.format(float(star[2]))],
                                                            ['min_separation', '{:.3f}'.format(float(star[3]))],
                                                            ['obs_window',
                                                             [star[4][0].datetime.strftime('%Y%m%d_%H%M%S.%f'),
                                                              star[4][1].datetime.strftime('%Y%m%d_%H%M%S.%f')]],
                                                            ['finding_chart', star[5]]
                                                            ]))

        return '_'.join(out_dict['name'].split(' ')), out_dict

    def set_mean_position(self, _epoch, _radec, _radec_dot=None, _mag=None):
        """
            Position at the middle of the night
        :param _epoch:
        :param _radec:
        :param _radec_dot:
        :param _mag:
        :return:
        """
        self.mean_position['epoch'] = _epoch
        self.mean_position['radec'] = _radec
        self.mean_position['radec_dot'] = _radec_dot
        self.mean_position['mag'] = _mag
        # # reset everything else if mean epoch is changed
        # self.meridian_crossing = None
        # self.is_observable = None
        # self.ephemeris = None
        # self.observability_windows = None
        # self.guide_stars = None

    def set_meridian_crossing(self, _meridian_crossing):
        self.meridian_crossing = _meridian_crossing

    def set_is_observable(self, _is_observable):
        self.is_observable = _is_observable

    def set_ephemeris(self, _ephemeris):
        self.ephemeris = _ephemeris

    def set_observability_windows(self, night, _elv_lim=45.0):

        # time grid within [0, 1] for interpolation:
        t = [ti.datetime for ti in self.ephemeris[:, 0]]
        t0 = t[0]
        t = np.array([(ti - t0).total_seconds() for ti in t])
        t /= t[-1]

        # interpolate to a denser grid to estimate periods when elv >= min_elv more precisely
        elv = map(float, self.ephemeris[:, -2])
        N_dense = 200
        t_dense = np.linspace(0, 1, N_dense)
        dense, _ = lagint(3, t, elv, t_dense)

        t_0 = self.ephemeris[0, 0]
        t_e = self.ephemeris[-1, 0]
        dt = t_e - t_0

        scans = []
        scan = []

        for t_d, el in zip(t_dense, dense):
            # print(t_d, el)
            # above elevation cut-off and the Sun is down?
            if el >= _elv_lim * np.pi / 180.0 and night[0] <= t_0 + t_d * dt <= night[1]:
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
            scans = t_0 + scans * dt

        self.observability_windows = scans

    def set_guide_stars(self, _jpl_eph, _station=None,
                        _radius=30.0, _margin=30.0, _m_lim_gs=16.0, eops=None):
        """ Get guide stars within radius arc seconds for each observability window.

        :param _jpl_eph:
        :param _station:
        :param _radius: maximum distance to guide star in arcseconds
        :param _margin: padd window with margin arcsec (one-sided margin)
        :param _m_lim_gs:
        :param _plot_field:
        :param _model_psf:
        :param _display_plot:
        :param _save_plot:
        :param _path_nightly_date:
        :param _cat_eop:
        :return:
        """
        global gaia

        if self.observability_windows is None:
            print('compute observability windows first before looking for guide stars')
            return False
        elif len(self.observability_windows) == 0 or not self.is_observable:
            print('target {:s} not observable'.format(self.object.name))
            return False
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
            # position of asteroid at arc start
            radec_start, _, vmag_start = self.object.raDecVmag(t_start.mjd, _jpl_eph, epoch='J2000',
                                                               station=_station, output_Vmag=True,
                                                               eops=eops)
            radec_start = np.array(radec_start)
            radec_start_sc = SkyCoord(ra=radec_start[0], dec=radec_start[1], unit=(u.rad, u.rad), frame='icrs')
            rdecra_start = np.hstack([1.0, radec_start[::-1]])
            # convert to cartesian
            start_cart = sph2cart(rdecra_start)
            # end of the 'arc'
            t_stop = window[1]
            # position of asteroid at arc end
            radec_stop, _, vmag_stop = self.object.raDecVmag(t_stop.mjd, _jpl_eph, epoch='J2000',
                                                             station=_station, output_Vmag=True,
                                                             eops=eops)
            radec_stop = np.array(radec_stop)
            radec_stop_sc = SkyCoord(ra=radec_stop[0], dec=radec_stop[1], unit=(u.rad, u.rad), frame='icrs')
            rdecra_stop = np.hstack([1.0, radec_stop[::-1]])
            stop_cart = sph2cart(rdecra_stop)

            arc_len = radec_stop_sc.separation(radec_start_sc)
            print('arc length: ', arc_len)

            # window time span
            window_t_span = t_stop - t_start

            # NOTE: asteroids don't move following great circles, and it's a bad idea to
            # approximate asteroid's motion with a great circle segment
            # Instead, we will compute distances from each of the grid stars to the asteroid actual positions
            # and search for minimum over the arc, then do a local fit to find the moment of closest approach
            # and compute time window and reference position using that.

            ''' compute distances from stars, return those (bright ones) that can be used as guide stars '''
            # print(t_start, t_stop, window_t_span)

            # first, 'split' the trajectory into separate positions that are ~radius_rad/2 apart
            t_step = (radius_rad / 2.0 * window_t_span) / arc_len.rad
            n_positions = int(np.ceil(arc_len.rad / radius_rad)) * 2
            print('split asteroid trajectory into {:d} pointings'.format(n_positions))
            radecs = []
            radec_dots = []
            for ii in range(n_positions):
                ti = t_start + t_step * ii
                rd, rddot, _ = self.object.raDecVmag(ti.mjd, _jpl_eph, epoch='J2000',
                                                     station=_station, output_Vmag=False,
                                                     eops=eops)
                radecs.append(rd)
                radec_dots.append(rddot)
            # print(radecs)
            radecs = np.array(radecs)
            radec_dots = np.array(radec_dots)  # these are in "/s
            # mean velocity along track:
            # v_along_track_mean = np.sqrt(np.mean(radec_dots[:, 0]) ** 2 + np.mean(radec_dots[:, 1]) ** 2)
            # print('v_along_track_mean:', v_along_track_mean)
            # print(arc_len.deg/3600.0 / window_t_span.sec)

            # trajectory midpoint:
            # no, no, no, asteroids don't do great circles :(
            # middle, radec_middle = great_circle_segment_midpoint(radec_start_sc, radec_stop_sc)
            # print('middle from great circle:', middle, radec_middle)
            t_middle = t_start + t_step * (n_positions // 2)
            radec_middle, _, _ = self.object.raDecVmag(t_middle.mjd, _jpl_eph, epoch='J2000',
                                                       station=_station, output_Vmag=False,
                                                       eops=eops)
            middle = SkyCoord(ra=radec_middle[0], dec=radec_middle[1], unit=(u.rad, u.rad), frame='icrs')
            # print('middle actual:', middle, radec_middle)

            # 'FoV' size + margins at both sides:
            ra_size = SkyCoord(ra=radec_start[0], dec=0, unit=(u.rad, u.rad), frame='icrs'). \
                separation(SkyCoord(ra=radec_stop[0], dec=0, unit=(u.rad, u.rad), frame='icrs')).rad
            dec_size = SkyCoord(ra=0, dec=radec_start[1], unit=(u.rad, u.rad), frame='icrs'). \
                separation(SkyCoord(ra=0, dec=radec_stop[1], unit=(u.rad, u.rad), frame='icrs')).rad
            window_size = np.array([ra_size, dec_size]) + 2.0 * np.array([_margin * np.pi / 180.0 / 3600,
                                                                          _margin * np.pi / 180.0 / 3600])

            # search large 'quadrant' around trajectory midpoint for "smaller" fields (<0.5 degree)
            if arc_len.deg < 0.5:

                # in arcsec:
                print('window_size in \":', window_size * 180.0 / np.pi * 3600)

                # do a 'global' search
                # viz.column_filters = {'<Gmag>': '<{:.1f}'.format(_m_lim_gs)}
                # grid_stars = viz.query_region(target, width=window_size[0] * u.rad, height=window_size[1] * u.rad,
                #                               catalog=_guide_star_cat)
                # print(middle)

                if True:
                    grid_stars = gaia.cone_search(middle, np.max(window_size) * u.rad).get_results()
                    # print(list(grid_stars['phot_g_mean_mag']))
                    if (grid_stars is None) or (len(list(grid_stars.keys())) == 0) or len(grid_stars) == 0:
                        # no stars found? proceed to next window
                        continue
                if False:
                    q = {"query_type": "cone_search",
                         "object_coordinates": {"radec": "[({:f}, {:f})]".format(middle.ra.deg, middle.dec.deg),
                                                "cone_search_radius": "{:f}".format(np.max(window_size)),
                                                "cone_search_unit": "rad"},
                         "catalogs": {"Gaia_DR1": {"filter": {},
                                                   "projection": {"source_id": 1,
                                                                  "ra": 1,
                                                                  "dec": 1,
                                                                  "phot_g_mean_mag": 1
                                                                  }
                                                   }
                                      }
                         }
                    grid_stars = kowalski.query_sync(query=q)
                    print(grid_stars)
            else:
                # for large fields, 'split' the trajectory and search in a smaller field around each 'pointing'
                # along the trajectory.
                # iterate over pointings, discard duplicates (if any)
                tables = []
                # global viz
                # print(pointings)
                max_pointings = 100
                if len(radecs) > max_pointings:
                    print('Moves very fast, will only consider the first {:d} pointings'.format(max_pointings))
                pointings = SkyCoord(ra=radecs[:, 0], dec=radecs[:, 1], unit=(u.rad, u.rad), frame='icrs')

                for pi, pointing in enumerate(pointings[:max_pointings]):
                    print('querying pointing #{:d}'.format(pi + 1))

                    if True:
                        # viz.column_filters = {'<Gmag>': '<{:.1f}'.format(_m_lim_gs)}
                        # grid_stars_pointing = viz.query_region(pointing, radius_rad * u.rad, catalog=_guide_star_cat,
                        #                                        cache=False)
                        grid_stars_pointing = gaia.cone_search(pointing, radius_rad * u.rad).get_results()
                        # print(grid_stars_pointing)
                        if (grid_stars_pointing is not None) and len(list(grid_stars_pointing.keys())) != 0:
                            print('number of stars in this pointing:', len(grid_stars_pointing))
                            if len(grid_stars_pointing) > 0:
                                tables.append(grid_stars_pointing)
                            # no stars found? proceed to next pointing

                    if False:
                        q = {"query_type": "cone_search",
                             "object_coordinates": {"radec": "[({:f}, {:f})]".format(pointing.ra.deg, pointing.dec.deg),
                                                    "cone_search_radius": "{:f}".format(radius_rad),
                                                    "cone_search_unit": "rad"},
                             "catalogs": {"Gaia_DR1": {"filter": {},
                                                       "projection": {"source_id": 1,
                                                                      "ra": 1,
                                                                      "dec": 1,
                                                                      "phot_g_mean_mag": 1
                                                                      }
                                                       }
                                          }
                             }
                        grid_stars = kowalski.query_sync(query=q)
                        print(grid_stars)

                print('number of pointings with stars:', len(tables))
                if len(tables) == 1:
                    grid_stars = tables[0]
                elif len(tables) > 1:
                    grid_stars = table.vstack(tables)
                    grid_stars = table.unique(grid_stars, keys='source_id')
                    # print(grid_stars)
                else:
                    # no stars? proceed to next obs. window
                    continue

            # print(grid_stars)

            # guide star magnitudes:
            grid_star_mags = np.array(grid_stars['phot_g_mean_mag'])
            # those that are bright enough for tip-tilt:
            mag_mask = grid_star_mags <= _m_lim_gs

            print('{:d} stars brighter than {:.3f}'.format(len(grid_stars[mag_mask]), _m_lim_gs))

            for star in grid_stars[mag_mask]:
                # grab a star by the coordinates :)
                radec_star = np.array([star['ra'], star['dec']]) * np.pi / 180.0  # [rad]
                # print('radec_star', radec_star)

                # astropy crd:
                # star_sc = SkyCoord(ra=radec_star[0], dec=radec_star[1], unit=(u.rad, u.rad), frame='icrs')

                separations = np.array([great_circle_distance(radec_star[1], radec_star[0], radec[1], radec[0])
                                        for radec in radecs])

                # close enough?
                if np.min(separations) < radius_rad:
                    print('{:s}:'.format(self.object.name), star['source_id'], 'is close enough!')

                    index_min, _, index_min_interpolated, distance_track = find_min(separations)
                    t_approach_star = t_start + t_step * index_min_interpolated
                    # get relevant derivatives ([almost] at the time of closest encounter):
                    radec_dot = radec_dots[index_min]
                    # asteroid apparent velocity along track ["/s] around t_closest_encounter:
                    v_along_track = np.sqrt(radec_dot[0] ** 2 + radec_dot[1] ** 2)

                    # arc length from closest point to star on star track (extension)
                    # to furthest such that distance to it <= radius_rad:
                    arc = np.arccos(np.cos(radius_rad) / np.cos(distance_track))
                    # print('arc len ": ', arc*180/np.pi*3600)

                    t_span = (arc / arc_len.rad) * window_t_span

                    # TODO: do not add if it's too short. need to be clever to check t_span vs asteroid mag
                    # shorter than 10 seconds? skip...
                    if t_span.sec * 2 < 20:
                        print('window time span shorter than 20 seconds, skipping')
                        continue

                    # tracking start and stop times
                    t_start_star = np.max([t_start, t_approach_star - t_span])
                    t_stop_star = np.min([t_stop, t_approach_star + t_span])

                    # save:
                    # name [RA, Dec] min_distance_in_arcsec [window_start_time, window_stop_time], finding_chart_png
                    self.guide_stars.append([star['source_id'], [star['ra'], star['dec']], star['phot_g_mean_mag'],
                                             distance_track * 180.0 / np.pi * 3600, [t_start_star, t_stop_star],
                                             'not available'])

                    # if False:
                    #     self.plot_field(middle, window_size=window_size,
                    #                     radec_start=radec_start, vmag_start=vmag_start,
                    #                     radec_stop=radec_stop, vmag_stop=vmag_stop,
                    #                     exposure=t_stop-t_start,
                    #                     _model_psf=_model_psf, grid_stars=grid_stars,
                    #                     _highlight_brighter_than_mag=_m_lim_gs,
                    #                     _display_plot=True)

        # keep 3 closest and 3 brightest, plot 'finding charts'
        if len(self.guide_stars) > 0:
            print('found {:d} suitable guide stars'.format(len(self.guide_stars)))
            self.guide_stars = np.array(self.guide_stars)

            dist = self.guide_stars[:, 3]
            # print(dist)
            mags = self.guide_stars[:, 2]
            # print(mags)

            grid_stars_closest = self.guide_stars[dist.argsort(), :][:3, :]
            # print(grid_stars_closest)
            grid_stars_brightest = self.guide_stars[mags.argsort(), :][:3, :]
            # print(grid_stars_brightest)
            best_grid_stars = grid_stars_brightest
            # print(best_grid_stars.shape)
            for star in grid_stars_closest:
                if star[0] not in best_grid_stars[:, 0]:
                    best_grid_stars = np.append(best_grid_stars, np.expand_dims(star, 0), axis=0)
                    # print(best_grid_stars.shape)

            self.guide_stars = best_grid_stars
            # print(self.guide_stars.shape)

        print('\t number of possible guide stars: {:d}'.format(len(self.guide_stars)))

        return True

    def plot_guide_stars(self, _jpl_eph, _guide_star_cat=u'I/337/gaia', _station=None,
                         _margin=30.0, _m_lim_gs=16.0, _model_psf=None, _display_plot=False,
                         _save_plot=False, _path_nightly_date='./', eops=None):
        """
            Plot fields with guide stars:
        :param _jpl_eph:
        :param _guide_star_cat:
        :param _station:
        :param _margin:
        :param _m_lim_gs:
        :param _model_psf:
        :param _display_plot:
        :param _save_plot:
        :param _path_nightly_date:
        :param _cat_eop:
        :return:
        """
        for si, star in enumerate(self.guide_stars):
            try:
                # print(star)
                star_name, star_radec, star_mag, star_min_dist, star_obs_window, _ = star
                t_start_star = star_obs_window[0]
                t_stop_star = star_obs_window[1]
                # print(star_name, star_radec, star_mag, star_min_dist, star_obs_window)

                radec_start_star, _, vmag_start_star = self.object.raDecVmag(t_start_star.mjd, _jpl_eph,
                                                                             epoch='J2000',
                                                                             station=_station,
                                                                             output_Vmag=True,
                                                                             eops=eops)
                radec_start_star = np.array(radec_start_star)
                # end of the 'arc'
                radec_stop_star, _, vmag_stop_star = self.object.raDecVmag(t_stop_star.mjd, _jpl_eph,
                                                                           epoch='J2000',
                                                                           station=_station,
                                                                           output_Vmag=True,
                                                                           eops=eops)
                radec_stop_star = np.array(radec_stop_star)

                # want to center on the track instead?
                # # first convert RA/Dec's on a unit sphere to Cartesian coordinates:
                # rdecra_start_star = np.hstack([1.0, radec_start_star[::-1]])
                # rdecra_stop_star = np.hstack([1.0, radec_stop_star[::-1]])
                # start_cart_star = sph2cart(rdecra_start_star)
                # stop_cart_star = sph2cart(rdecra_stop_star)
                #
                # # middle point for the 'FoV':
                # lamb = 1 + np.dot(start_cart_star, stop_cart_star) / 1.0 ** 2
                # middle_cart_star = (start_cart_star + stop_cart_star) / np.sqrt(2.0 * lamb)
                # rdecra_middle_star = cart2sph(middle_cart_star)
                # radec_middle_star = rdecra_middle_star[:-3:-1]

                # 'FoV' + margins:
                ra_size_star = SkyCoord(ra=radec_start_star[0], dec=0, unit=(u.rad, u.rad), frame='icrs'). \
                    separation(SkyCoord(ra=radec_stop_star[0], dec=0, unit=(u.rad, u.rad), frame='icrs')).rad
                dec_size_star = SkyCoord(ra=0, dec=radec_start_star[1], unit=(u.rad, u.rad), frame='icrs'). \
                    separation(SkyCoord(ra=0, dec=radec_stop_star[1], unit=(u.rad, u.rad), frame='icrs')).rad
                window_size_star = np.array([ra_size_star, dec_size_star]) + 2.0 * np.array(
                    [_margin * np.pi / 180.0 / 3600, _margin * np.pi / 180.0 / 3600])

                # Robo-AO VIC FoV is 36"x36":
                # window_size_star = np.array([36 * np.pi / 180.0 / 3600, 36 * np.pi / 180.0 / 3600])
                # in arcsec:
                # print('window_size in \":', window_size_star * 180.0 / np.pi * 3600)

                # window time span
                window_t_span_star = t_stop_star - t_start_star

                # stars in the field (without mag cut-off):
                star_sc = SkyCoord(ra=star_radec[0], dec=star_radec[1], unit=(u.deg, u.deg), frame='icrs')
                # fov_stars = viz.query_region(star_sc, width=window_size_star[0] * u.rad,
                #                              height=window_size_star[1] * u.rad,
                #                              catalog=_guide_star_cat, cache=False)
                fov_stars = gaia.query_object(star_sc, width=window_size_star[0] * u.rad,
                                              height=window_size_star[1] * u.rad)

                # center plot on star:
                print('plotting', self.object.name, star_name, star_obs_window[0], star_obs_window[1])
                plot_name = '_'.join(self.object.name.split(' ')) + '__' + \
                            '_'.join(str(star_name).split(' '))
                self.plot_field(star_sc, window_size=window_size_star,
                                radec_start=radec_start_star, vmag_start=vmag_start_star,
                                radec_stop=radec_stop_star, vmag_stop=vmag_stop_star,
                                exposure=window_t_span_star,
                                _model_psf=_model_psf, grid_stars=fov_stars,
                                _highlight_brighter_than_mag=_m_lim_gs,
                                _display_plot=_display_plot, _save_plot=_save_plot,
                                path=_path_nightly_date, name=plot_name)
                # save the name if succesful:
                self.guide_stars[si, -1] = '{:s}.png'.format(plot_name)
            except Exception as e:
                print(e)
                traceback.print_exc()

    @staticmethod
    def plot_field(target, window_size, radec_start, vmag_start, radec_stop, vmag_stop,
                   exposure, _model_psf, grid_stars, _highlight_brighter_than_mag=16.0, scale_bar_size=20,
                   _display_plot=False, _save_plot=False, path='./', name='field'):
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
        pix_stars = np.array(w.wcs_world2pix(grid_stars['ra'], grid_stars['dec'], 0)).T
        # apply linear + SIP:
        # pix_stars = np.array(w.all_world2pix(grid_stars['_RAJ2000'], grid_stars['_DEJ2000'], 0)).T
        # pix_stars = np.array(w.all_world2pix(grid_stars['RA_ICRS'], grid_stars['DE_ICRS'], 0)).T
        mag_stars = np.array(grid_stars['phot_g_mean_mag'])
        # print(pix_stars)
        # print(mag_stars)

        # tic = _time()
        # asteroid_start = coord.SkyCoord(ra=radec_start[0], dec=radec_start[1], unit=(u.rad, u.rad), frame='icrs')
        # asteroid_stop = coord.SkyCoord(ra=radec_stop[0], dec=radec_stop[1], unit=(u.rad, u.rad), frame='icrs')
        radec_start_deg = radec_start * 180.0 / np.pi
        radec_stop_deg = radec_stop * 180.0 / np.pi

        # check if start or stop are outside FoV, correct if necessary? no, just plot the full window

        pix_asteroid = np.array([w.wcs_world2pix(radec_start_deg[0], radec_start_deg[1], 0),
                                 w.wcs_world2pix(radec_stop_deg[0], radec_stop_deg[1], 0)])
        mag_asteroid = np.mean([vmag_start, vmag_stop])
        # print(pix_asteroid, pix_asteroid.shape)
        # print(mag_asteroid)
        # print(_time() - tic)

        # tic = _time()
        sim_image = generate_image(xy=pix_stars, mag=mag_stars,
                                   xy_ast=pix_asteroid, mag_ast=mag_asteroid, exp=exposure.sec,
                                   nx=w.naxis1, ny=w.naxis2, psf=_model_psf)
        # print(_time() - tic)

        # tic = _time()
        # convert simulated image to fits hdu:
        hdu = fits.PrimaryHDU(sim_image, header=w.to_header())

        ''' plot! '''
        plt.close('all')
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
        fig.show_markers(radec_start_deg[0], radec_start_deg[1],
                         layer='marker_set_1', edgecolor=plt.cm.Blues(0.2),
                         facecolor=plt.cm.Blues(0.3), marker='o', s=50, alpha=0.7, linewidths=3)
        fig.show_markers(radec_stop_deg[0], radec_stop_deg[1],
                         layer='marker_set_2', edgecolor=plt.cm.Oranges(0.5),
                         facecolor=plt.cm.Oranges(0.3), marker='x', s=50, alpha=0.7, linewidths=3)

        # highlight stars bright enough to serve as tip-tilt guide stars:
        mask_bright = mag_stars <= _highlight_brighter_than_mag
        if np.max(mask_bright) == 1:
            fig.show_markers(grid_stars[mask_bright]['ra'], grid_stars[mask_bright]['dec'],
                             layer='marker_set_2', edgecolor=plt.cm.Oranges(0.9),
                             facecolor=plt.cm.Oranges(0.8), marker='+', s=50, alpha=0.9, linewidths=1)

        # fig.show_arrows([asteroid_start.ra.rad*u.rad], [asteroid_start.dec.rad*u.rad],
        #                 [np.sign(pix_asteroid[1, 0] - pix_asteroid[0, 0]) *
        #                 coord.SkyCoord(ra=radec_start[0], dec=0, unit=(u.rad, u.rad), frame='icrs').
        #                  separation(coord.SkyCoord(ra=radec_stop[0], dec=0, unit=(u.rad, u.rad), frame='icrs'))],
        #                 [np.sign(pix_asteroid[1, 1] - pix_asteroid[0, 1]) *
        #                 coord.SkyCoord(ra=0, dec=radec_start[1], unit=(u.rad, u.rad), frame='icrs').
        #                  separation(coord.SkyCoord(ra=0, dec=radec_stop[1], unit=(u.rad, u.rad), frame='icrs'))],
        #                 edgecolor=plt.cm.Oranges(0.5), facecolor=plt.cm.Oranges(0.3))

        # add scale bar
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


class TargetList(object):
    """
        Produce (nightly) target list for the queue
    """

    def __init__(self, _f_inp, database_source='mpc', database_file=None, _observatory='kpno',
                 _m_lim=16.0, _elv_lim=40.0, _date=None, timezone='America/Phoenix'):
        # init database
        if database_source == 'mpc':
            if database_file is None:
                db = AsteroidDatabaseMPC()
            else:
                if database_file != 'CometEls.txt':
                    db = AsteroidDatabaseMPC(_f_database=database_file)
                else:
                    db = CometDatabaseMPC(_f_database=database_file)
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
        if _date is None:
            _now = datetime.datetime.now(pytz.timezone(timezone))
            _date = datetime.datetime(_now.year, _now.month, _now.day) + datetime.timedelta(days=1)

        self.date = _date

        # load pypride stuff
        self.eops = load_eops(_cat_eop=self.inp['cat_eop'], _date=_date)

        if _observatory == 'kpno':
            # site: KPNO 2.1 m: 31.958182 N, -111.598273 W, 2086.7 m (31d57'29.5"N 111d35'53.8"W)
            r_GTRS = geodetic2Ecef(lat=31.958182 * np.pi / 180.0, lon=-111.598273 * np.pi / 180.0, h=2086.7e-3)
            r_GTRS = np.array(r_GTRS) * 1e3  # in meters
            self.sta = Site(r_GTRS)
        else:
            print('Station is not Kitt Peak! Falling back to geocentric ra/decs')
            self.sta = Site([0, 0, 0])

        ''' targets '''
        self.targets = np.array([], dtype=object)
        # self.targets = []

    @staticmethod
    def azels(tt, eops, jpl_eph, vw, lon_gcen, u, v, r_GTRS, radecs):
        """
            ra, dec - this is for moving objects, in general. therefore, they must be arrays of the same size as tt

            return in radians
        """
        ''' set coordinates '''
        # K_s = np.array([np.cos(dec) * np.cos(ra),
        #                 np.cos(dec) * np.sin(ra),
        #                 np.sin(dec)])

        # source vectors:
        K_s = np.vstack([np.cos(radecs[:, 1]) * np.cos(radecs[:, 0]),
                         np.cos(radecs[:, 1]) * np.sin(radecs[:, 0]),
                         np.sin(radecs[:, 1])])

        azels = []
        azels_sun = []

        for ti, t in enumerate(tt):
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
            CT, dTAIdCT = t_eph(JD, UT1, TT, lon_gcen, u, v)

            ''' rotation matrix IERS '''
            r2000 = GTRS_to_GCRS(tstamp, eop_int, dTAIdCT)

            ''' do not compute displacements due to geophysical effects '''

            ''' get position and velocity in GCRS '''
            r_GCRS = np.dot(r2000[:, :, 0], r_GTRS)
            v_GCRS = np.dot(r2000[:, :, 1], r_GTRS)

            ''' earth position '''
            rrd = pleph(JD + CT, 3, 12, jpl_eph)
            earth = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3

            ''' sun position '''
            rrd = pleph(JD + CT, 11, 12, jpl_eph)
            sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
            r = sun[:, 0] - (r_GCRS + earth[:, 0])
            ra = np.arctan2(r[1], r[0])  # right ascension
            dec = np.arctan(r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2))  # declination
            if ra < 0:
                ra += 2.0 * np.pi
            K_s_sun = np.array([np.cos(dec) * np.cos(ra),
                                np.cos(dec) * np.sin(ra),
                                np.sin(dec)])

            # refraction formula:
            # from 'U. S. Naval Observatory\'s Vector Astrometry Software':
            # 1.02 * (1.0 / np.tan(0 + 10.3 / (0 + 5.11)))

            az, el = aber_source(v_GCRS, vw, K_s[:, ti], r2000, earth)
            az_sun, el_sun = aber_source(v_GCRS, vw, K_s_sun, r2000, earth)

            azels.append([az, el])
            azels_sun.append([az_sun, el_sun])

        return np.array(azels), np.array(azels_sun)

    def observability(self, times, radecs, Vmags, twilight='nautical', fraction=0.1):
        """
            Check target observability
        Parameters
        ----------
        times : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)
        radecs : object RA and Dec s
        twilight:
        fraction:

        Returns
        -------
        ret1 : bool if target observable or not during the night
        ret2 : bool if target crosses the meridian or not during the night
        ret3 : azels for the night
        """
        # if not isinstance(time, Time):
        #     time = Time(time)
        # times = self._generate_24hr_grid(time[0], 0, 1, N, for_deriv=False)

        # r_GTRS = np.array(self.observatory.location.value)
        azels, azels_sun = self.azels(times, self.eops, self.inp['jpl_eph'], self.sta.vw, self.sta.lon_gcen,
                                      self.sta.u, self.sta.v, self.sta.r_GTRS, radecs)
        altitudes = np.array(azels)[:, 1]
        altitudes_sun = np.array(azels_sun)[:, 1]
        # print(np.array(zip(times.iso, altitudes*180/np.pi)))
        # print('\n')

        # to check meridian crossing when above the horizon
        t = np.linspace(0, 1, len(times))
        # dense grid with source elevations:
        t_dense = np.linspace(0, 1, 200)
        # p = np.polyfit(t, altitudes, 6)
        # altitudes_dense = np.polyval(p, t_dense)
        altitudes_dense, _ = lagint(3, t, altitudes, t_dense)

        maxp = np.argmax(altitudes_dense)
        maxv = np.max(altitudes_dense)
        root = maxp / float(len(t_dense))
        # minus = np.polyval(p, maxp - 0.01)
        # plus = np.polyval(p, maxp + 0.01)
        # dve shagi nalevo, dve shagi napravo:
        minus, _ = lagint(3, t_dense, altitudes_dense, root - 0.01)
        plus, _ = lagint(3, t_dense, altitudes_dense, root + 0.01)
        # print(times[0], root * u.day, times[-1], maxv, minus, plus)

        # when above the horizon!
        crosses_meridian = (times[0] + root * u.day < times[-1]) and (np.max((minus, plus)) < maxv)

        # to check observability:
        # dense grid with sun elevations:
        # p_sun = np.polyfit(t, altitudes_sun, 6)
        # altitudes_sun_dense = np.polyval(p_sun, t_dense)
        altitudes_sun_dense, _ = lagint(3, t, altitudes_sun, t_dense)

        # densify Vmags too:
        # p_Vmag = np.polyfit(t, Vmags, 6)
        # Vmags_dense = np.polyval(p_Vmag, t_dense)
        Vmags_dense, _ = lagint(3, t, Vmags, t_dense)

        if twilight == 'astronomical':
            sun_elv_lim = -18  # deg
        elif twilight == 'civil':
            sun_elv_lim = -6
        elif twilight == 'nautical':
            sun_elv_lim = -12
        else:
            sun_elv_lim = -12

        # fig = plt.figure()
        # ax = fig.add_subplot(311)
        # ax.plot(t_dense, altitudes_dense * 180/np.pi)
        # ax = fig.add_subplot(312)
        # ax.plot(t_dense, altitudes_sun_dense * 180/np.pi)
        # ax = fig.add_subplot(313)
        # ax.plot(t_dense, Vmags_dense)
        # plt.show()

        # condition checks:
        observability_array = np.all(np.vstack([altitudes_dense >= self.elv_lim * np.pi / 180.0,
                                                altitudes_sun_dense <= sun_elv_lim * np.pi / 180.0,
                                                Vmags_dense <= self.m_lim]), axis=0)
        # print(self.elv_lim, sun_elv_lim, self.m_lim, np.sum(observability_array), float(len(observability_array)),
        #       np.sum(observability_array)/float(len(observability_array)) >= fraction)

        # np.sum counts True values in observability_array
        observable = np.sum(observability_array) / float(len(observability_array)) >= fraction

        # print('pypride ', (time[0] + root * u.day).iso)

        # print(observable, crosses_meridian)
        # raw_input()

        # return bool if observable, crosses the meridian and array with source azimuths and elevations
        return observable, crosses_meridian, azels

    def target_list(self, _date, _mask=None, _parallel=False, _epoch='J2000', _output_Vmag=True, _night_grid_n=50,
                    _twilight='nautical', _fraction=0.1):
        """
            Get observational parameters for a (masked) target list from self.database for bright enough targets
            Check whether targets are observable and store only those
            Implement in subclass
        """
        raise NotImplementedError

    def get_obs_params(self, target, _mjd, epoch='J2000', output_Vmag=True):
        """ Compute obs parameters for a given t

        """
        raise NotImplementedError

    def get_night(self, _date, night_grid_n=50):
        """
            _date - datetime.datetime object, 0h UTC of the coming day
        """
        nextDay = _date + datetime.timedelta(days=1)
        astrot = Time([_date.strftime('%Y-%m-%d %H:%M:%S.%f'), nextDay.strftime('%Y-%m-%d %H:%M:%S.%f')],
                      format='iso', scale='utc')
        # when the night comes, heh?
        sunSet = self.observatory.sun_set_time(astrot[0])
        sunRise = self.observatory.sun_rise_time(astrot[1])

        # print(sunSet.datetime[0].strftime('%Y-%m-%d %H:%M:%S.%f'),
        # sunRise.datetime[0].strftime('%Y-%m-%d %H:%M:%S.%f'))
        night = Time([sunSet.datetime[0].strftime('%Y-%m-%d %H:%M:%S.%f'),
                      sunRise.datetime[0].strftime('%Y-%m-%d %H:%M:%S.%f')],
                     format='iso', scale='utc')

        night_len = (night[1] - night[0]).sec / 86400.0  # in days
        time_grid = np.linspace(0, night_len, night_grid_n) * u.day

        return night, night[0] + time_grid

    def get_observing_windows(self, day):
        """
            Get observing windows for when the targets in the target list are above elv_lim
        :return:
        """
        if len(self.targets) == 0:
            print('no targets in the target list')
            return

        # print(self.targets)

        # get night:
        night, _ = self.get_night(day)

        for tn, target in enumerate(self.targets):
            if target.is_observable:
                self.targets[tn].set_observability_windows(night=night, _elv_lim=self.elv_lim)

    def get_guide_stars(self, _guide_star_cat=u'I/337/gaia', _radius=30.0, _margin=30.0, _m_lim_gs=16.0,
                        _plot_field=False, _psf_fits=None, _display_plot=False, _save_plot=False,
                        _path_nightly_date='./', parallel=False):
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

        if parallel:
            # FIXME: can't pickle fortran object error :(
            # targ_args = [self.inp['jpl_eph'], _guide_star_cat, self.observatory_pypride,
            #              _radius, _margin, _m_lim_gs, _plot_field, model_psf]
            # args = [[target] + targ_args for target in self.targets]
            # futures = client.map(target_guide_star_helper, args)
            # self.targets = client.gather(futures)
            pass
        else:
            for tn, target in enumerate(self.targets):
                if target.is_observable:
                    status = self.targets[tn].set_guide_stars(_jpl_eph=self.inp['jpl_eph'],
                                                              _station=deepcopy(self.sta), _radius=_radius, _margin=_margin,
                                                              _m_lim_gs=_m_lim_gs, eops=self.eops)
                    if status and _plot_field:
                        self.targets[tn].plot_guide_stars(_jpl_eph=self.inp['jpl_eph'],
                                                          _station=deepcopy(self.sta), _margin=_margin,
                                                          _m_lim_gs=_m_lim_gs, _model_psf=model_psf,
                                                          _display_plot=_display_plot, _save_plot=_save_plot,
                                                          _path_nightly_date=_path_nightly_date,
                                                          eops=self.eops)

    def make_nightly_json(self, _path_nightly_date='./'):
        """
            Create json file with bright objects
        :param _path_nightly_date:
        :return:
        """

        json_dict = OrderedDict([tr.to_dict() for tr in self.targets])
        # print(json_dict)

        # for tr in self.targets:
        #     print(tr)

        # print(os.path.join(_path_nightly_date, '{:s}.json'.format('bright_objects')))
        if not os.path.exists(_path_nightly_date):
            os.makedirs(_path_nightly_date)
        with open(os.path.join(_path_nightly_date, '{:s}.json'.format('bright_objects')), 'w') as fj:
            json.dump(json_dict, fj, indent=4)  # sort_keys=True,


class TargetListAsteroids(TargetList):
    """
        Produce (nightly) target list for the asteroids project
    """

    def __init__(self, _f_inp, database_source='mpc', database_file=None, _observatory='kpno',
                 _m_lim=16.0, _elv_lim=40.0, _date=None, timezone='America/Phoenix'):

        # simply init super class
        TargetList.__init__(self, _f_inp=_f_inp, database_source=database_source, database_file=database_file,
                            _observatory=_observatory, _m_lim=_m_lim, _elv_lim=_elv_lim, _date=_date, timezone=timezone)

    def target_list(self, _date, _mask=None, _parallel=False, _epoch='J2000',
                    _output_Vmag=True, _night_grid_n=50, _twilight='nautical', _fraction=0.1):
        """
            Get observational parameters for a (masked) target list
            from self.database
        """
        # get middle of night:
        night, night_grid = self.get_night(_date, _night_grid_n)
        mjds = night_grid.tdb.mjd  # in TDB!!

        # iterate over asteroids:
        target_list = []
        if _mask is None:
            _mask = range(0, len(self.database))

        # do it in parallel
        if _parallel:
            ttic = _time()
            n_cpu = multiprocessing.cpu_count()
            # create pool
            pool = multiprocessing.Pool(n_cpu)
            # asynchronously apply target_list_helper
            database_masked = self.database[_mask]
            args = [(self, asteroid, mjds, _epoch, _output_Vmag, night_grid, _twilight, _fraction)
                    for asteroid in database_masked]
            result = pool.map_async(target_list_helper, args)
            # close bassejn
            pool.close()  # we are not adding any more processes
            pool.join()  # wait until all threads are done before going on
            # get the ordered results
            targets_all = result.get()
            target_list = [_t for _t in targets_all if _t is not None]
            target_list = np.array(target_list)
            print('parallel computation took: {:.2f} s'.format(_time() - ttic))
        else:
            ttic = _time()
            for ia, asteroid in enumerate(self.database[_mask]):
                print(asteroid['name'])
                tic = _time()
                # target can move fast, so compute radec/dots on a grid for the whole night
                radecs, radec_dots, Vmags = [], [], []
                for mjd in mjds:
                    radec, radec_dot, Vmag = self.get_obs_params(asteroid, mjd, _epoch, _output_Vmag)
                    radecs.append(radec)
                    radec_dots.append(radec_dot)
                    Vmags.append(Vmag)
                radecs = np.array(radecs)
                radec_dots = np.array(radec_dots)
                Vmags = np.array(Vmags)

                print(len(self.database) - ia, _time() - tic)
                # skip if too dim
                if np.min(Vmags) <= self.m_lim:
                    observable, meridian_transit, azels = self.observability(night_grid, radecs, Vmags,
                                                                             twilight=_twilight, fraction=_fraction)
                    target_list.append([MinorBodyDatabase.init_minor_body(asteroid), observable, meridian_transit,
                                        radecs, radec_dots, azels, Vmags])
            target_list = np.array(target_list)
            print('serial computation took: {:.2f} s'.format(_time() - ttic))

        print('Total targets brighter than {:.1f}: {:d}'.format(self.m_lim, target_list.shape[0]))
        if target_list.shape[0] > 0:
            print('Total observable targets: {:d}'.format(np.count_nonzero(target_list[:, 1])))

        # set up target objects for self.targets:
        for entry in target_list:
            # print(entry[0].name, entry[1])
            target = Target(_object=entry[0])
            target.set_is_observable(entry[1])
            target.set_meridian_crossing(entry[2])
            target.set_ephemeris(np.hstack([np.expand_dims(night_grid, 1), entry[3],
                                            entry[4], entry[5], np.expand_dims(entry[6], 1)]))

            self.targets = np.append(self.targets, target)
            # self.targets.append(target)
            # print('targets:', self.targets)

    def get_obs_params(self, target, _mjd, epoch='J2000', output_Vmag=True):
        """ Compute obs parameters for a given t

        :param target: Kepler class object
        :param _mjd: epoch in TDB/mjd (t.tdb.mjd, t - astropy.Time object, UTC)
        :param epoch: 'J2000', 'Date' or jdate like 2017.0
        :param output_Vmag: compute Vmag?
        :return: radec in rad, radec_dot in arcsec/s, Vmag
        """

        AU_DE430 = 1.49597870700000000e+11  # m
        GSUN = 0.295912208285591100e-03 * AU_DE430 ** 3 / 86400.0 ** 2
        # convert AU to m:
        a = target['a'] * AU_DE430
        e = target['e']
        # convert deg to rad:
        i = target['i'] * np.pi / 180.0
        w = target['w'] * np.pi / 180.0
        Node = target['Node'] * np.pi / 180.0
        M0 = target['M0'] * np.pi / 180.0
        t0 = target['epoch']
        H = target['H']
        G = target['G']

        asteroid = MinorBody(target['name'], a=a, e=e, i=i, w=w, Node=Node, M0=M0, GM=GSUN, t0=t0, H=H, G=G)

        # jpl_eph - path to eph used by pypride
        radec, radec_dot, Vmag = asteroid.raDecVmag(_mjd, self.inp['jpl_eph'], station=deepcopy(self.sta),
                                                    epoch=epoch, output_Vmag=output_Vmag,
                                                    eops=self.eops)

        return radec, radec_dot, Vmag


class TargetListComets(TargetList):
    """
        Produce (nightly) target list for the comets project
    """

    def __init__(self, _f_inp, database_source='mpc', database_file='CometEls.txt', _observatory='kpno',
                 _m_lim=16.0, _elv_lim=40.0, _date=None, timezone='America/Phoenix'):

        # simply init super class
        TargetList.__init__(self, _f_inp=_f_inp, database_source=database_source, database_file=database_file,
                            _observatory=_observatory, _m_lim=_m_lim, _elv_lim=_elv_lim,
                            _date=_date, timezone=timezone)

    def target_list(self, _date, _mask=None, _parallel=False, _epoch='J2000',
                    _output_Vmag=True, _night_grid_n=50, _twilight='nautical', _fraction=0.1):
        """
            Get observational parameters for a (masked) target list
            from self.database
        """
        # get middle of night:
        night, night_grid = self.get_night(_date, _night_grid_n)
        mjds = night_grid.tdb.mjd  # in TDB!!

        # iterate over asteroids:
        target_list = []
        if _mask is None:
            _mask = range(0, len(self.database))

        # do it in parallel
        if _parallel:
            ttic = _time()
            n_cpu = multiprocessing.cpu_count()
            # create pool
            pool = multiprocessing.Pool(n_cpu)
            # asynchronously apply target_list_helper
            database_masked = self.database[_mask]
            args = [(self, comet, mjds, _epoch, _output_Vmag, night_grid, _twilight, _fraction)
                    for comet in database_masked]
            result = pool.map_async(target_list_helper, args)
            # close bassejn
            pool.close()  # we are not adding any more processes
            pool.join()  # wait until all threads are done before going on
            # get the ordered results
            targets_all = result.get()
            target_list = [_t for _t in targets_all if _t is not None]
            target_list = np.array(target_list)
            print('parallel computation took: {:.2f} s'.format(_time() - ttic))
        else:
            ttic = _time()
            for ia, comet in enumerate(self.database[_mask]):
                print(comet['name'])
                tic = _time()
                # target can move fast, so compute radec/dots on a grid for the whole night
                radecs, radec_dots, Vmags = [], [], []
                for mjd in mjds:
                    radec, radec_dot, Vmag = self.get_obs_params(comet, mjd, _epoch, _output_Vmag)
                    radecs.append(radec)
                    radec_dots.append(radec_dot)
                    Vmags.append(Vmag)
                radecs = np.array(radecs)
                radec_dots = np.array(radec_dots)
                Vmags = np.array(Vmags)

                print(len(self.database) - ia, _time() - tic)
                # skip if too dim
                if np.min(Vmags) <= self.m_lim:
                    observable, meridian_transit, azels = self.observability(night_grid, radecs, Vmags,
                                                                             twilight=_twilight, fraction=_fraction)
                    target_list.append([MinorBodyDatabase.init_minor_body(comet), observable, meridian_transit,
                                        radecs, radec_dots, azels, Vmags])
            target_list = np.array(target_list)
            print('serial computation took: {:.2f} s'.format(_time() - ttic))

        print('Total targets brighter than {:.1f}: {:d}'.format(self.m_lim, target_list.shape[0]))
        if target_list.shape[0] > 0:
            print('Total observable targets: {:d}'.format(np.count_nonzero(target_list[:, 1])))

        # set up target objects for self.targets:
        for entry in target_list:
            # print(entry[0].name, entry[1])
            target = Target(_object=entry[0])
            target.set_is_observable(entry[1])
            target.set_meridian_crossing(entry[2])
            target.set_ephemeris(np.hstack([np.expand_dims(night_grid, 1), entry[3],
                                            entry[4], entry[5], np.expand_dims(entry[6], 1)]))

            self.targets = np.append(self.targets, target)
            # self.targets.append(target)
            # print('targets:', self.targets)

    def get_obs_params(self, target, _mjd, epoch='J2000', output_Vmag=True):
        """ Compute obs parameters for a given t

        :param target: Kepler class object
        :param _mjd: epoch in TDB/mjd (t.tdb.mjd, t - astropy.Time object, UTC)
        :param epoch: 'J2000', 'Date' or jdate like 2017.0
        :param output_Vmag: compute Vmag?
        :return: radec in rad, radec_dot in arcsec/s, Vmag
        """

        AU_DE430 = 1.49597870700000000e+11  # m
        GSUN = 0.295912208285591100e-03 * AU_DE430 ** 3 / 86400.0 ** 2
        # convert AU to m:
        q = target['q'] * AU_DE430
        e = target['e']
        # convert deg to rad:
        i = target['i'] * np.pi / 180.0
        w = target['w'] * np.pi / 180.0
        Node = target['Node'] * np.pi / 180.0
        t0 = target['epoch']
        tau = target['tau']
        H = target['H']
        G = target['G']

        comet = MinorBody(target['name'], q=q, e=e, i=i, w=w, Node=Node, tau=tau, GM=GSUN, t0=t0, H=H, G=G)

        # jpl_eph - path to eph used by pypride
        radec, radec_dot, Vmag = comet.raDecVmag(_mjd, self.inp['jpl_eph'], station=deepcopy(self.sta),
                                                 epoch=epoch, output_Vmag=output_Vmag, eops=self.eops)

        # FIXME: debugging output
        # print(target['name'], [Angle(radec[0], unit=u.rad).hms, Angle(radec[1], unit=u.rad).dms],
        #       radec_dot, Vmag, Time(_mjd, format='mjd', scale='tdb').utc.datetime)

        return radec, radec_dot, Vmag


class MinorBodyDatabase(object):
    def __init__(self):
        self.database = None
        self.f_database = None
        self.path_local = None
        self.database_url = None

    def minor_body_database_update(self, n=1.0):
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
                    print('MinorBody database: {:s} is out of date, updating...'.format(self.f_database))
            else:
                do_update = True
                print('Database file: {:s} is missing, fetching...'.format(self.f_database))
            # if the file is older than n days:
            if do_update:
                try:
                    response = urllib2.urlopen(self.database_url)
                    with open(self.f_database, 'w') as f:
                        f.write(response.read())
                    print('Done.')
                except Exception as err:
                    print(str(err))
                    traceback.print_exc()
                    pass

    @staticmethod
    def init_minor_body(_minor_body_db_entry):
        """
            Initialize MinorBody object from 'raw' db entry
        """
        _kep_el = _minor_body_db_entry.dtype.names

        AU_DE430 = 1.49597870700000000e+11  # m
        GSUN = 0.295912208285591100e-03 * AU_DE430 ** 3 / 86400.0 ** 2
        # convert AU to m:
        a = _minor_body_db_entry['a'] * AU_DE430 if 'a' in _kep_el else None
        q = _minor_body_db_entry['q'] * AU_DE430 if 'q' in _kep_el else None
        e = _minor_body_db_entry['e']
        # convert deg to rad:
        i = _minor_body_db_entry['i'] * np.pi / 180.0
        w = _minor_body_db_entry['w'] * np.pi / 180.0
        Node = _minor_body_db_entry['Node'] * np.pi / 180.0
        M0 = _minor_body_db_entry['M0'] * np.pi / 180.0 if 'M0' in _kep_el else None
        tau = _minor_body_db_entry['tau'] * np.pi / 180.0 if 'tau' in _kep_el else None
        t0 = _minor_body_db_entry['epoch']
        H = _minor_body_db_entry['H']
        G = _minor_body_db_entry['G']

        return MinorBody(_minor_body_db_entry['name'], a=a, q=q, e=e, i=i, w=w, Node=Node,
                         M0=M0, GM=GSUN, t0=t0, H=H, G=G)

    def load(self):
        """
            Load database into self.database
        :return:
        """
        raise NotImplementedError

    def get_one(self, _name):
        """
            Get one asteroid from database
        :return: single MinorBody object
        """
        raise NotImplementedError

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of MinorBody objects
        """
        raise NotImplementedError


class AsteroidDatabaseJPL(MinorBodyDatabase):
    def __init__(self, _path_local='./', _f_database='ELEMENTS.NUMBR'):
        # initialize super class
        super(AsteroidDatabaseJPL, self).__init__()

        self.f_database = _f_database
        self.path_local = _path_local
        self.database_url = os.path.join('http://ssd.jpl.nasa.gov/dat/', _f_database)

        # update database if necessary:
        self.minor_body_database_update(n=1.0)

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
        :return: single MinorBody object
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
                # initialize and return MinorBody object here:
                if len(asteroid_entry) > 0:
                    return self.init_minor_body(asteroid_entry)
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

                dt = np.dtype([('name', '|S21'),
                               ('epoch', '<i8'), ('a', '<f8'),
                               ('e', '<f8'), ('i', '<f8'),
                               ('w', '<f8'), ('Node', '<f8'),
                               ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
                # this is for numbered asteroids:
                asteroid_entry = np.array([((entry[0:25].strip(),) + tuple(map(float, entry[25:].split()[:-2])))],
                                          dtype=dt)
                # initialize and return MinorBody object here:
                return self.init_minor_body(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of MinorBody objects
        """
        if self.database is None:
            self.load()

        asteroids = [self.init_minor_body(asteroid_entry) for asteroid_entry in self.database]
        return asteroids


class AsteroidDatabaseMPC(MinorBodyDatabase):
    def __init__(self, _path_local='./', _f_database='PHA.txt'):
        super(AsteroidDatabaseMPC, self).__init__()

        self.f_database = _f_database
        self.path_local = _path_local
        self.database_url = os.path.join('http://www.minorplanetcenter.net/iau/MPCORB/', _f_database)

        # update database if necessary:
        self.minor_body_database_update(n=0.5)

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
        :return: single MinorBody object
        """
        # remove underscores:
        if '_' in _name:
            _name = ' '.join(_name.split('_'))

        if self.database is not None:
            try:
                asteroid_entry = self.database[self.database['name'] == _name]
                # initialize and return MinorBody object here:
                if len(asteroid_entry) > 0:
                    return self.init_minor_body(asteroid_entry)
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
                # initialize MinorBody object here:
                return self.init_minor_body(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of MinorBody objects
        """
        if self.database is None:
            self.load()

        minor_bodies = [self.init_minor_body(asteroid_entry) for asteroid_entry in self.database]
        return minor_bodies


class CometDatabaseMPC(MinorBodyDatabase):
    def __init__(self, _path_local='./', _f_database='CometEls.txt'):
        super(MinorBodyDatabase, self).__init__()

        self.f_database = _f_database
        self.path_local = _path_local
        self.database_url = os.path.join('http://www.minorplanetcenter.net/iau/MPCORB/', _f_database)

        # update database if necessary:
        self.minor_body_database_update(n=0.5)

    def load(self):
        """
            Load database into self.database
        :return:
        """
        with open(os.path.join(self.path_local, self.f_database), 'r') as f:
            database = f.readlines()

        # remove empty lines:
        database = [l.replace('\t', ' ') for l in database if len(l) > 5]

        dt = np.dtype([('name', '|S21'), ('year_per_pass', '<i8'), ('month_per_pass', '<i8'),
                       ('day_per_pass', '<f8'), ('tau', '<f8'), ('q', '<f8'), ('e', '<f8'),
                       ('w', '<f8'), ('Node', '<f8'), ('i', '<f8'),
                       ('epoch', '<f8'), ('H', '<f8'), ('G', '<f8'),
                       ('designation', '|S21'), ('reference', '|S21')
                       ])

        self.database = []
        for entry in database:
            _name = str(entry[:14]).strip().replace(' ', '')
            _year_per_pass = int(entry[14:19])
            _month_per_pass = int(entry[19:21])
            _day_per_pass = float(entry[21:28])
            _tau = mjuliandate(_year_per_pass, _month_per_pass, _day_per_pass)
            _q = float(entry[30:39])
            _e = float(entry[41:49])
            _w = float(entry[51:59])
            _Node = float(entry[61:69])
            _i = float(entry[71:79])
            _solution_epoch = datetime.datetime.strptime(str(entry[81:89]), '%Y%m%d') \
                if len(str(entry[81:89].strip())) > 0 else datetime.datetime(_year_per_pass,
                                                                             _month_per_pass,
                                                                             int(_day_per_pass))
            _epoch = Time(_solution_epoch.strftime('%Y-%m-%d %H:%M:%S.%f'), format='iso', scale='tt').mjd
            _H = float(entry[91:95])
            _G = float(entry[96:100])
            _designation = str(entry[102:158].strip())
            _reference = str(entry[159:188].strip())
            # print(_name, _year_per_pass, _month_per_pass, _day_per_pass, _q, _e, _w, _Node,
            #       _i, _solution_epoch, _epoch, _H, _G, _designation, _reference)
            # raw_input()

            self.database.append((_name, _year_per_pass, _month_per_pass,
                                  _day_per_pass, _tau, _q, _e, _w, _Node,
                                  _i, _epoch, _H, _G, _designation, _reference))

        self.database = np.array(self.database, dtype=dt)
        # print(self.database)
        # raw_input()

    def get_one(self, _name):
        """
            Get one asteroid from database
        :return: single MinorBody object
        """
        # remove underscores:
        if '_' in _name:
            _name = ' '.join(_name.split('_'))

        if self.database is not None:
            try:
                asteroid_entry = self.database[self.database['name'] == _name]
                # initialize and return MinorBody object here:
                if len(asteroid_entry) > 0:
                    return self.init_minor_body(asteroid_entry)
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

                dt = np.dtype([('name', '|S21'), ('year_per_pass', '<i8'), ('month_per_pass', '<i8'),
                               ('day_per_pass', '<f8'), ('tau', '<f8'), ('q', '<f8'), ('e', '<f8'),
                               ('w', '<f8'), ('Node', '<f8'), ('i', '<f8'),
                               ('epoch', '<f8'), ('H', '<f8'), ('G', '<f8'),
                               ('designation', '|S21'), ('reference', '|S21')
                               ])

                _name = str(entry[:14]).strip().replace(' ', '')
                _year_per_pass = int(entry[14:19])
                _month_per_pass = int(entry[19:21])
                _day_per_pass = float(entry[21:28])
                _tau = mjuliandate(_year_per_pass, _month_per_pass, _day_per_pass)
                _q = float(entry[30:39])
                _e = float(entry[41:49])
                _w = float(entry[51:59])
                _Node = float(entry[61:69])
                _i = float(entry[71:79])
                _solution_epoch = datetime.datetime.strptime(str(entry[81:89]), '%Y%m%d')
                _epoch = Time(_solution_epoch.strftime('%Y-%m-%d %H:%M:%S.%f'), format='iso', scale='tt').mjd
                _H = float(entry[91:95])
                _G = float(entry[96:100])
                _designation = str(entry[102:158].strip())
                _reference = str(entry[159:188].strip())

                asteroid_entry = np.array((_name, _year_per_pass, _month_per_pass,
                                           _day_per_pass, _tau, _q, _e, _w, _Node,
                                           _i, _epoch, _H, _G, _designation, _reference), dtype=dt)
                # initialize MinorBody object here:
                return self.init_minor_body(asteroid_entry)

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None

    def get_all(self):
        """
            Get all asteroids from database
        :return: list of MinorBody objects
        """
        if self.database is None:
            self.load()

        minor_bodies = [self.init_minor_body(asteroid_entry) for asteroid_entry in self.database]
        return minor_bodies


class MinorBody(object):
    """
       Class to work with Keplerian orbits of Minor Bodies
    """

    def __init__(self, name, e, i, w, Node, GM, t0, a=None, M0=None, q=None, tau=None, H=None, G=None):
        """

        :param name: 
        :param e: 
        :param i: 
        :param w: 
        :param Node: 
        :param GM: 
        :param t0: 
        :param a: 
        :param M0:
        :param q:
        :param tau: 
        :param H: 
        :param G:

            For asteroids, a and M0 are given
            For comets, q and tau are given
        """
        # a should be in metres, all the angles - in radians, t0 in mjd [days]
        # GM = G*(M_central_body + M_body) for 2 body problem
        # GM = G*M_central_body**3/ (M_central_body + M_body)**2
        #                   for restricted 3 body problem
        # GM = [m**2/kg/s**2]
        self.name = name
        self.e = float(e)
        self.i = float(i)
        self.w = float(w)
        self.Node = float(Node)
        self.a = float(a) if a is not None else None
        self.M0 = float(M0) if M0 is not None else None
        self.q = float(q) if q is not None else None
        self.tau = float(tau) if tau is not None else None
        self.GM = float(GM)
        self.t0 = float(t0)
        self.H = float(H)
        self.G = float(G)

        # print(name, e, i, w, Node, GM, t0, a, M0, q, tau, H, G)

    def __str__(self):
        """
            Print it out nicely
        """
        if self.a is not None:
            return '<Keplerian object {:s}: a={:e} m, e={:f}, i={:f} rad, '. \
                       format(self.name, self.a, self.e, self.i) + \
                   'w={:f} rad, Node={:f} rad, M0={:f} rad, '. \
                       format(self.w, self.Node, self.M0) + \
                   't0={:f} (MJD), GM={:e} m**3/kg/s**2>'. \
                       format(self.t0, self.GM)
        else:
            return '<Keplerian object {:s}: q={:e} m, e={:f}, i={:f} rad, '. \
                       format(self.name, self.q, self.e, self.i) + \
                   'w={:f} rad, Node={:f} rad, tau={:f} (MJD), '. \
                       format(self.w, self.Node, self.tau) + \
                   't0={:f} (MJD), GM={:e} m**3/kg/s**2>'. \
                       format(self.t0, self.GM)

    @staticmethod
    @jit
    def kepler(e, M, max_iter=10):
        """ Solve Kepler's equation for elliptical motion

        :param e: eccentricity
        :param M: mean anomaly, rad
        :return:
        """
        E = deepcopy(M)
        tmp = 1e5
        n_iter = 1

        while (np.abs(E - tmp) > 1e-9) or (n_iter <= max_iter):
            tmp = deepcopy(E)
            E += (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
            n_iter += 1

        return E

    @staticmethod
    @jit
    def kepler_hyp(e, M, max_iter=10):
        """ Solve Kepler's equation for hiperbolic motion

        :param e: eccentricity
        :param M: mean anomaly, rad
        :return:
        """
        H = deepcopy(M)
        Q = (H + M) / e
        tmp_H = 1e5
        # tmp_Q = 1e5
        n_iter = 1

        # while ((np.abs(H - tmp_H) > 1e-9) and (np.abs(Q - tmp_Q) > 1e-9)) or (n_iter <= max_iter):
        while (np.abs(H - tmp_H) > 1e-9) or (n_iter <= max_iter):
            tmp_H = deepcopy(H)
            # tmp_Q = deepcopy(Q)
            H = np.log(np.sqrt(Q ** 2 + 1) + Q)
            Q = (H + M) / e
            n_iter += 1

        return H

    @jit
    def to_cart(self, t):
        """
            Compute Cartesian state at epoch t with respect to the central body
            from Keplerian elements
            t -- epoch in mjd [decimal days]
        """
        # print(self.name, self.e, self.a, self.q, self.tau, self.M0)

        if (0.0 < self.e < 1.0) and (self.a is not None) and (self.M0 is not None):
            ''' elliptical motion: (asteroids) '''
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
            p = self.a * (1.0 - self.e ** 2)

        elif (0.0 < self.e < 1.0) and (self.q is not None) and (self.tau is not None):
            ''' elliptical motion: (comets) '''
            a = self.q / (1.0 - self.e)
            # mean motion:
            n = np.sqrt(self.GM / a / a / a) * 86400.0  # [rad/day]
            # mean anomaly at t:
            M = n * (t - self.tau)
            #        print(np.fmod(M, 2*np.pi))
            # solve Kepler equation, get eccentric anomaly:
            E = self.kepler(self.e, M)
            cosE = np.cos(E)
            sinE = np.sin(E)
            # get true anomaly and distance from focus:
            sinv = np.sqrt(1.0 - self.e ** 2) * sinE / (1.0 - self.e * cosE)
            cosv = (cosE - self.e) / (1.0 - self.e * cosE)
            r = a * (1.0 - self.e ** 2) / (1.0 + self.e * cosv)
            #        r = self.a*(1 - self.e*cosE)
            p = a * (1.0 - self.e ** 2)

        elif (self.e == 1.0) and (self.q is not None) and (self.tau is not None):  # to machine precision
            ''' parabolic motion '''
            # mean motion:
            n = np.sqrt(self.GM / (2.0 * self.q))
            # focal parameter:
            p = (4.0 * self.GM / (n ** 2)) ** (1.0 / 3.0)
            # mean anomaly at t given pericenter passage epoch:
            M = n * (t - self.tau)
            Q = 2.0 / 3.0 * M
            R = (np.sqrt(1.0 + Q ** 2) + np.abs(Q)) ** (2.0 / 3.0)
            S = 3.0 * M / (R + 1.0 + 1.0 / R)
            # get true anomaly and distance from focus:
            sinv = 2.0 * S / (1.0 + S ** 2)
            cosv = (1.0 - S ** 2) / (1.0 + S ** 2)
            r = p * (1 + cosv)
            # r = p*(1.0 + S**2) / 2

        elif (self.e > 1.0) and (self.q is not None) and (self.tau is not None):
            ''' hyperbolic motion '''
            # real semi axis
            a = self.q / (self.e - 1.0)
            # mean motion:
            n = np.sqrt(self.GM / a / a / a) * 86400.0  # [rad/day]
            # mean anomaly at t given pericenter passage epoch:
            M = n * (t - self.tau)
            # solve Kepler equation's analogue
            H = self.kepler_hyp(self.e, M)
            coshH = np.cosh(H)
            sinhH = np.sinh(H)
            # get true anomaly and distance from focus:
            sinv = np.sqrt(self.e ** 2 - 1.0) * sinhH / (self.e * coshH - 1.0)
            cosv = (self.e - coshH) / (self.e * coshH - 1.0)
            r = a * (self.e ** 2 - 1.0) / (1.0 + self.e * cosv)
            # r = a*(self.e*coshH - 1.0)
            p = a * (self.e ** 2 - 1.0)

        else:
            raise Exception('Bad set of Keplerian elements, cannot proceed: {:s}'.format(self.name))

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
    # @memoize
    def PNmatrix(t, lon_gcen, u, v, eops):
        """
            Compute (geocentric) IAU2000 precession/nutation matrix for epoch
            t -- astropy.time.Time object
        """
        # precess to date
        ''' set dates: '''

        tstamp = t.datetime
        mjd = np.floor(t.mjd)
        UTC = (tstamp.hour + tstamp.minute / 60.0 + tstamp.second / 3600.0) / 24.0
        JD = mjd + 2400000.5

        ''' compute tai & tt '''
        TAI, TT = taitime(mjd, UTC)

        ''' load cats '''
        # don't load it every freakin time
        # eops = load_eops(_cat_eop, tstamp)

        ''' interpolate eops to tstamp '''
        UT1, eop_int = eop_iers(mjd, UTC, eops)

        ''' compute coordinate time fraction of CT day at GC '''
        CT, dTAIdCT = t_eph(JD, UT1, TT, lon_gcen, u, v)

        ''' rotation matrix IERS '''
        r2000 = GTRS_to_GCRS(tstamp, eop_int, dTAIdCT)
        #    print(r2000[:,:,0])

        return r2000

    # for some reason, jit-compilation of this one slows down the execution significantly!
    def raDecVmag(self, mjd, jpl_eph, epoch='J2000', station=None, output_Vmag=False, eops=None):
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
        # print(state)

        # station GCRS position/velocity:
        # if (station is not None) and (station.r_GCRS is None):
        if station is not None:
            r2000 = self.PNmatrix(Time(mjd, format='mjd'), station.lon_gcen, station.u, station.v, eops)
            station.GTRS_to_GCRS(r2000)

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

            if self.tau is None:
                # asteroid?
                Vmag = self.H - 2.5 * np.log10((1.0 - self.G) * phi1 + self.G * phi2) + \
                       5.0 * np.log10(EA_norm * SA_norm / AU_DE430 ** 2)
            else:
                # comet? compute total brightness:
                Vmag = self.H + 5.0 * np.log10(EA_norm / AU_DE430) + 2.5 * self.G * np.log10(SA_norm / AU_DE430)

            ''' Comets:
            m1 = H + 5 log (delta) + 2.5G log (r)
            delta: geocentric distance
            r: heliocentric distance

             T-mag N-mag =
               Comet's approximate apparent visual total magnitude ("T-mag") and nuclear
            magnitude ("N-mag") by following standard IAU definitions:
                T-mag =  M1 + 5*log10(delta) + k1*log10(r)
                N-mag =  M2 + 5*log10(delta) + k2*log10(r) + phcof*beta
               Units: MAGNITUDES
            '''

        # returning SkyCoord is handy, but very expensive
        # return (SkyCoord(ra=ra, dec=dec, unit=(u.rad, u.rad), frame='icrs'),
        #         (ra_dot, dec_dot), Vmag)
        return [ra, dec], [ra_dot, dec_dot], Vmag


def getModefromMag(mag):
    """
        VICD mode depending on the object magnitude
    """
    m = float(mag)
    if m < 8:
        mode = '6'
    elif 8 <= m < 10:
        mode = '7'
    elif 10 <= m < 12:
        mode = '8'
    elif 12 <= m < 13:
        mode = '9'
    elif m >= 13:
        mode = '10'
    return mode


class TargetXML(object):
    """
        Class to handle queue target xml files
    """

    def __init__(self, path, program_number, server='http://localhost:8081'):
        self.program_number = int(program_number)
        self.path = os.path.join(path, 'Program_{:d}'.format(int(program_number)))
        self.server = server
        self.Targets = None

    def getAllTargetXML(self):
        """
            check if there are target XML files under self.path
            load them if affirmative
        """
        nXML = len([f for f in os.listdir(self.path)
                    if 'Target_' in f and f[0] != '.'])
        if nXML == 0:
            return None
        else:
            targets = {}
            for f in os.listdir(self.path):
                if 'Target_' in f and f[0] != '.':
                    tree = ET.parse(os.path.join(self.path, f))
                    root = tree.getroot()

                    # Ordnung muss sein!
                    targ = OrderedDict()
                    for content in root:
                        if content.tag != 'Object':
                            targ[content.tag] = content.text
                        else:
                            if 'Object' not in targ:
                                targ['Object'] = []
                            obj = OrderedDict()
                            for data_obj in content:
                                if data_obj.tag != 'Observation':
                                    obj[data_obj.tag] = data_obj.text
                                else:
                                    if 'Observation' not in obj:
                                        obj['Observation'] = []
                                    obs = OrderedDict()
                                    for data_obs in data_obj:
                                        obs[data_obs.tag] = data_obs.text
                                    obj['Observation'].append(obs)
                            targ['Object'].append(obj)

                    targets[f] = targ

        self.Targets = targets

    def getTargetNames(self):
        """
            Get target names
        """
        if self.Targets is None:
            return None
        else:
            targetNames = {self.Targets[t]['name']: t for t in self.Targets}

        return targetNames

    @staticmethod
    def dummyXML(program_number=-1, obj='asteroid'):
        """

        :param program_number:
        :param obj: 'asteroid' or 'ploon'
        :return:
        """

        if obj == 'asteroid':
            return OrderedDict([('program_number', program_number),
                                ('number', ''),
                                ('name', ''),
                                ('visited_times_for_completion', 3),
                                # ('seeing_limit', ''),
                                ('visited_times', 0),
                                ('done', 0),
                                ('cadence', 0),
                                ('comment', 'None'),
                                ('time_critical_flag', 0),
                                ('Object',
                                 [OrderedDict([('number', 1),
                                               ('RA', ''),
                                               ('dec', ''),
                                               ('ra_rate', ''),
                                               ('dec_rate', ''),
                                               ('epoch', '2000.0'),
                                               ('magnitude', ''),
                                               ('solar_system', 1),
                                               # ('sun_altitude_limit', ''),
                                               # ('moon_phase_window', ''),
                                               # ('airmass_limit', ''),
                                               # ('sun_distance_limit', ''),
                                               # ('moon_distance_limit', ''),
                                               # ('sky_brightness_limit', ''),
                                               # ('hour_angle_limit', ''),
                                               ('done', 0),
                                               ('Observation',
                                                [OrderedDict([('number', 1),
                                                              ('exposure_time', 180),
                                                              ('ao_flag', 1),
                                                              ('filter_code', 'FILTER_SLOAN_I'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)])])])])])
        else:
            return OrderedDict([('program_number', program_number),
                                ('number', ''),
                                ('name', ''),
                                ('visited_times_for_completion', 300),
                                # ('seeing_limit', ''),
                                ('visited_times', 0),
                                ('done', 0),
                                ('cadence', 0),
                                ('comment', 'None'),
                                ('time_critical_flag', 0),
                                ('Object',
                                 [OrderedDict([('number', 1),
                                               ('RA', ''),
                                               ('dec', ''),
                                               ('ra_rate', ''),
                                               ('dec_rate', ''),
                                               ('epoch', '2000.0'),
                                               ('magnitude', ''),
                                               ('solar_system', 1),
                                               # ('sun_altitude_limit', ''),
                                               # ('moon_phase_window', ''),
                                               # ('airmass_limit', ''),
                                               # ('sun_distance_limit', ''),
                                               # ('moon_distance_limit', ''),
                                               # ('sky_brightness_limit', ''),
                                               # ('hour_angle_limit', ''),
                                               ('done', 0),
                                               ('Observation',
                                                [OrderedDict([('number', 1),
                                                              ('exposure_time', 30),
                                                              ('ao_flag', 1),
                                                              ('filter_code', 'FILTER_SLOAN_I'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)]),
                                                 OrderedDict([('number', 2),
                                                              ('exposure_time', 30),
                                                              ('ao_flag', 1),
                                                              ('filter_code', 'FILTER_SLOAN_Z'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)]),
                                                 OrderedDict([('number', 3),
                                                              ('exposure_time', 30),
                                                              ('ao_flag', 1),
                                                              ('filter_code', 'FILTER_SLOAN_R'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)]),
                                                 OrderedDict([('number', 4),
                                                              ('exposure_time', 30),
                                                              ('ao_flag', 1),
                                                              ('filter_code', 'FILTER_SLOAN_G'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)])
                                                 ])])])])

    def dumpTargets(self, targets, epoch='J2000'):
        """ Dump target list

        :param targets:
        :param epoch:
        :param _server:
        :return:
        """
        # load existing target xml data:
        self.getAllTargetXML()
        # get their names:
        targetNames = self.getTargetNames()
        max_xml_num = max([int(l[l.index('_') + 1:l.index('.xml')])
                           for l in targetNames.values()]) \
            if targetNames is not None else 0
        target_max_num = max([int(self.Targets[t]['number'])
                              for t in self.Targets]) \
            if self.Targets is not None else 0

        # iterate over targets
        added_target_xml_files = 0
        for target in targets:
            # if not is_planet_or_moon(target[0]['name']):
            # asteroid or planet?
            if not isinstance(target[0], dict) and 'num' in target[0].dtype.names:
                name = '{:d} {:s}'.format(target[0]['num'], target[0]['name'])
            else:
                name = '{:s}'.format(target[0]['name'])

            # no spaces, please :(
            name = name.replace(' ', '_')
            # no primes too, please :(
            name = name.replace('\'', '_')

            # update existing xml file
            if targetNames is not None and name in targetNames:
                # print(name)
                xml = self.Targets[targetNames[name]]

                if is_planet_or_moon(name) or not is_multiple_asteroid(name):
                    xml['comment'] = 'modified_{:s}'.format('_'.join(str(datetime.datetime.now()).split()))
                elif is_multiple_asteroid(name):
                    xml['comment'] = \
                        'known_multiple;_modified_{:s}'.format('_'.join(str(datetime.datetime.now()).split()))
                xml['Object'][0]['RA'] = \
                    '{:02.0f}:{:02.0f}:{:02.3f}'.format(*hms(target[2][0]))
                dec = dms(target[2][1])
                xml['Object'][0]['dec'] = \
                    '{:02.0f}:{:02.0f}:{:02.3f}'.format(dec[0], abs(dec[1]), abs(dec[2]))
                ''' !!! NOTE: ra rate must be with a minus sign !!! '''
                xml['Object'][0]['ra_rate'] = '{:.5f}'.format(-target[3][0])
                ''' NOTE 2017/03/15: not sure about the new TCS! will try as is '''
                xml['Object'][0]['ra_rate'] = '{:.5f}'.format(target[3][0])
                xml['Object'][0]['dec_rate'] = '{:.5f}'.format(target[3][1])
                #                print target[1].decimalyear, target[1].jyear,
                #                        2000.0 + (target[1].jd-2451544.5)/365.25
                if epoch == 'J2000':
                    xml['Object'][0]['epoch'] = '{:.1f}'.format(2000.0)
                else:
                    xml['Object'][0]['epoch'] = '{:.9f}'.format(target[1].jyear)
                xml['Object'][0]['magnitude'] = '{:.3f}'.format(target[4])
                # set hour angle limit if target crosses meridian during the night:
                if target[-1]:
                    xml['Object'][0]['hour_angle_limit'] = '0.5'
                else:
                    xml['Object'][0]['hour_angle_limit'] = ''
                # planet or moon?
                if is_planet_or_moon(name):
                    # set up correct filters:
                    # xml['Object'][0]['Observation'][0]['filter_code'] = 'FILTER_SLOAN_I'
                    # since we want to observe them every night,
                    # we need to force the queue to do so
                    xml['done'] = 0
                    xml['Object'][0]['done'] = 0
                    for ii, _ in enumerate(xml['Object'][0]['Observation']):
                        xml['Object'][0]['Observation'][ii]['done'] = 0
                for ii, _ in enumerate(xml['Object'][0]['Observation']):
                    xml['Object'][0]['Observation'][ii]['camera_mode'] = \
                        '{:s}'.format(getModefromMag(target[4]))

                target_xml_path = os.path.join(self.path, targetNames[name])
            # print target_xml_path

            # create a new xml file
            else:
                if is_planet_or_moon(name):
                    obj = 'ploon'
                else:
                    obj = 'asteroid'
                xml = self.dummyXML(self.program_number, obj=obj)
                added_target_xml_files += 1

                xml['number'] = target_max_num + added_target_xml_files
                xml['name'] = name
                if is_planet_or_moon(name) or not is_multiple_asteroid(name):
                    xml['comment'] = 'modified_{:s}'.format('_'.join(str(datetime.datetime.now()).split()))
                elif is_multiple_asteroid(name):
                    xml['comment'] = \
                        'known_multiple;_modified_{:s}'.format('_'.join(str(datetime.datetime.now()).split()))
                xml['Object'][0]['RA'] = \
                    '{:02.0f}:{:02.0f}:{:02.3f}'.format(*hms(target[2][0]))
                dec = dms(target[2][1])
                xml['Object'][0]['dec'] = \
                    '{:02.0f}:{:02.0f}:{:02.3f}'.format(dec[0], abs(dec[1]), abs(dec[2]))
                ''' !!! NOTE: ra rate must be with a minus sign !!! '''
                xml['Object'][0]['ra_rate'] = '{:.5f}'.format(-target[3][0])
                xml['Object'][0]['dec_rate'] = '{:.5f}'.format(target[3][1])
                if epoch == 'J2000':
                    xml['Object'][0]['epoch'] = '{:.1f}'.format(2000.0)
                else:
                    xml['Object'][0]['epoch'] = '{:.9f}'.format(target[1].jyear)
                xml['Object'][0]['magnitude'] = '{:.3f}'.format(target[4])
                # set hour angle limit if target crosses meridian during the night:
                if target[-1]:
                    xml['Object'][0]['hour_angle_limit'] = '0.5'
                else:
                    xml['Object'][0]['hour_angle_limit'] = ''
                # planet or moon?
                if is_planet_or_moon(name):
                    # set up correct filters:
                    # xml['Object'][0]['Observation'][0]['filter_code'] = 'FILTER_SLOAN_I'
                    # since we want to observe them every night,
                    # we need to force the queue to do so
                    xml['done'] = 0
                    xml['Object'][0]['done'] = 0
                    for ii, _ in enumerate(xml['Object'][0]['Observation']):
                        xml['Object'][0]['Observation'][ii]['done'] = 0
                for ii, _ in enumerate(xml['Object'][0]['Observation']):
                    xml['Object'][0]['Observation'][ii]['camera_mode'] = \
                        '{:s}'.format(getModefromMag(target[4]))

                target_xml_path = os.path.join(self.path,
                                               'Target_{:d}.xml'.format(max_xml_num +
                                                                        added_target_xml_files))
            # print target_xml_path

            # build an xml-file:
            target_xml = dicttoxml(xml, custom_root='Target', attr_type=False)
            # this is good enough, but adds unnecessary <item> tags. remove em:
            dom = minidom.parseString(target_xml)
            target_xml = dom.toprettyxml()
            # <item>'s left extra \t's after them - remove them:
            target_xml = target_xml.replace('\t\t\t', '\t\t')
            target_xml = target_xml.replace('\t\t\t\t', '\t\t\t')
            target_xml = target_xml.replace('<?xml version="1.0" ?>', '')
            target_xml = target_xml.split('\n')
            # remove empty tags
            target_xml = [t for t in target_xml if 'item>' not in t \
                          and '/>' not in t]

            ind_obs_start = [i for i, v in enumerate(target_xml) if '<Observation>' in v]
            ind_obs_stop = [i for i, v in enumerate(target_xml) if '</Observation>' in v]
            for (start, stop) in zip(ind_obs_start, ind_obs_stop):
                ind_num_obs = [i + start for i, v in enumerate(target_xml[start:stop])
                               if '<number>' in v]
                if len(ind_num_obs) > 1:
                    for ind in ind_num_obs[:0:-1]:
                        target_xml.insert(ind, '\t\t</Observation>\n\t\t<Observation>')

            # print target_xml

            with open(target_xml_path, 'w') as f:
                for line in target_xml[1:-1]:
                    f.write('{:s}\n'.format(line))
                f.write('{:s}'.format(target_xml[-1]))

        # update Programs.xml if necessary
        try:
            r = requests.get(self.server, auth=('admin', 'robo@0'))
            if int(r.status_code) != 200:
                print('server error')
        except Exception:
            print('failed to connect to the website.')
            return 1

    def clean_target_list(self):
        """
            Remove targets from the queue if it doesn't satisfy
            observability criteria any more (and thus was not just updated)
        """
        pnot = len([_f for _f in os.listdir(self.path)
                    if 'Target_' in _f and _f[0] != '.'])
        # iterate over target xml files
        target_nums_to_remove = []

        target_list_xml = ['Target_{:d}.xml'.format(i + 1) for i in range(int(pnot))]

        for targ_num, target_xml in enumerate(target_list_xml):
            tree = ET.parse(os.path.join(self.path, target_xml))
            root = tree.getroot()

            targ = dict()
            targ['Object'] = []
            for content in root:
                if content.tag != 'Object':
                    targ[content.tag] = content.text
                else:
                    obj = dict()
                    obj['Observation'] = []
                    for data_obj in content:
                        if data_obj.tag != 'Observation':
                            obj[data_obj.tag] = data_obj.text
                        else:
                            obs = dict()
                            for data_obs in data_obj:
                                obs[data_obs.tag] = data_obs.text
                            obj['Observation'].append(obs)
                    targ['Object'].append(obj)

            try:
                # TODO: the queue software does not like spaces even in comments
                t_xml = datetime.datetime.strptime(' '.join(targ['comment'].split('_')[-2:]),
                                                   '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # why not?
                t_xml = datetime.datetime.now() - datetime.timedelta(days=10)
            # updated > 2 days ago?
            if (datetime.datetime.now() - t_xml).total_seconds() > 86400 * 2:
                # if (datetime.datetime.now() - t_xml).total_seconds() > 86400*2 \
                #         and targ['done'] == '0' \
                #         and targ['Object'][0]['Observation'][0]['repeated'] == '0':
                # print(targ['comment'], targ['done'])
                target_nums_to_remove.append(targ_num + 1)

        # now remove the xml files. start from end not to scoop numbering
        if len(target_nums_to_remove) > 0:
            try:
                for _targ_num in target_nums_to_remove[::-1]:
                    r = requests.get(os.path.join(self.server, 'removeTarget'),
                                     auth=('admin', 'robo@0'),
                                     params={'program_number': int(self.program_number),
                                             'target_number': int(_targ_num)})
                    # print(_targ_num, r.status_code)
                    if int(r.status_code) != 200:
                        print('server error')
                print('removed {:d} targets that are no longer suitable for observations.'
                      .format(len(target_nums_to_remove)))
                # call main page to modify/fix Programs.xml
                r = requests.get(self.server, auth=('admin', 'robo@0'))
                if int(r.status_code) != 200:
                    print('server error')
            except Exception:
                print('failed to remove targets via the website.')


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    parser.add_argument('--asteroid_class', metavar='asteroid_class', action='store', dest='asteroid_class',
                        help='asteroid class: nea, pha, daily, temporary', type=str, default='pha')

    args = parser.parse_args()
    config_file = args.config_file
    asteroid_class = args.asteroid_class

    if asteroid_class == 'pha':
        asteroid_database_file = 'PHA.txt'
    elif asteroid_class == 'nea':
        asteroid_database_file = 'NEA.txt'
    elif asteroid_class == 'daily':
        asteroid_database_file = 'DAILY.DAT'
    else:
        print('could not recognize asteroid class, falling back to PHA')
        asteroid_class = 'pha'
        asteroid_database_file = 'PHA.txt'

    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    # config.read(os.path.join(abs_path, 'config.ini'))
    config.read(os.path.join(abs_path, config_file))

    f_inp = config.get('Path', 'pypride_inp')
    path_nightly = config.get('Path', 'path_nightly')

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
    display_plot = eval(config.get('Vizier', 'display_plot'))
    save_plot = eval(config.get('Vizier', 'save_plot'))

    run_tests = False
    if run_tests:
        print('running tests...')
        # run tests:
        # asteroid name must be MPC-friendly
        # asteroid_name = '(3200)_Phaethon'
        asteroid_name = '2017_BQ6'

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

    # time_str = '20161022_042047.042560'
    # start_time = Time(str(datetime.datetime.strptime(time_str, '%Y%m%d_%H%M%S.%f')), format='iso', scale='utc')

    ''' Target list '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    # now = datetime.datetime.now(pytz.timezone(timezone))
    date0 = datetime.datetime.utcnow()
    # date0 = datetime.datetime(2017, 6, 1)

    for dd in range(0, 2):
        date = datetime.datetime(date0.year, date0.month, date0.day) + datetime.timedelta(days=dd)
        # date = datetime.datetime(2017, 1, 30)
        print('\nrunning computation for:', date)
        print('\nstarted at:', datetime.datetime.now(pytz.timezone(timezone)))

        # NEA or PHA:
        tl = TargetListAsteroids(f_inp, database_source='mpc', database_file=asteroid_database_file,
                                 _observatory=observatory, _m_lim=m_lim, _elv_lim=elv_lim, _date=date)
        # get all bright targets given m_lim and check observability given elv_lim, twilight and fraction
        mask = None
        tl.target_list(date, mask, _parallel=True, _epoch='J2000', _output_Vmag=True, _night_grid_n=40,
                       _twilight=twilight, _fraction=fraction)
        # get observing windows
        tl.get_observing_windows(date)
        # get guide stars
        path_nightly_date = os.path.join(path_nightly, date.strftime('%Y%m%d'), asteroid_class)
        tl.get_guide_stars(_guide_star_cat=guide_star_cat, _radius=radius, _margin=margin, _m_lim_gs=m_lim_gs,
                           _plot_field=plot_field, _psf_fits=psf_fits, _display_plot=display_plot,
                           _save_plot=save_plot, _path_nightly_date=path_nightly_date, parallel=False)
        # save json
        tl.make_nightly_json(_path_nightly_date=path_nightly_date)

        del tl, date
        # clear_download_cache()
        # force garbage collection:
        # gc.collect()

        # client.shutdown()
