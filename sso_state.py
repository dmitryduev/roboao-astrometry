# coding=utf-8
"""
Report

@autor: Dr Dmitry A. Duev [Caltech]


Run like this:
python sso_current_state.py -- '3_Juno' 14:53:23.481 -08:02:0.600 -0.00595 0.00129

* a double minus is used to tell the parser not to look for optional arguments
* not to confuse it with a possible negative ra/dec/[rate] value

"""

from __future__ import print_function
import argparse
import os
import numpy as np
from numba import jit
from copy import deepcopy
import linecache
from astropy.time import Time
import ephem
import datetime
import urllib2
import inspect
import traceback
import ConfigParser
from pypride.classes import inp_set
from pypride.vintflib import pleph
from pypride.vintlib import sph2cart, cart2sph, iau_PNM00A
from pypride.vintlib import eop_update, mjuliandate


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        # self.print_help(sys.stderr)
        # self.exit(2, '%s: error: %s\n' % (self.prog, message))
        # print error code 1 (not enough arguments) and exit
        self.exit(2, '1\n')


def dms(rad):
    d, m = divmod(abs(rad), np.pi/180)
    m, s = divmod(m, np.pi/180/60)
    s /= np.pi/180/3600
    if rad >= 0:
        return [d, m, s]
    else:
        return [-d, -m, -s]


def hms(rad):
    if rad < 0:
        rad += np.pi
    h, m = divmod(rad, np.pi/12)
    m, s = divmod(m, np.pi/12/60)
    s /= np.pi/12/3600
    return [h, m, s]


class Kepler(object):
    """
       Class to work with Keplerian orbits (of Asteroids)
    """

    def __init__(self, a, e, i, w, Node, M0, GM, t0, H=None, G=None):
        # a should be in metres, all the angles - in radians, t0 in mjd [days]
        # GM = G*(M_central_body + M_body) for 2 body problem
        # GM = G*M_central_body**3/ (M_central_body + M_body)**2
        #                   for restricted 3 body problem
        # GM = [m**2/kg/s**2]
        self.a = a
        self.e = e
        self.i = i
        self.w = w
        self.Node = Node
        self.M0 = M0
        self.GM = GM
        self.t0 = float(t0)
        self.H = H
        self.G = G

    def __str__(self):
        """
            Print it out nicely
        """
        return '<Keplerian object: a={:e} m, e={:f}, i={:f} rad, '. \
                   format(self.a, self.e, self.i) + \
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
            epsilon at J2000 = 23°.43929111 - from DE200
            ε = 23° 26′ 21″.406 − 46″.836769 T −
                0″.0001831 T**2 + 0″.00200340 T**3 −
                0″.576×10−6 T**4 − 4″.34×10−8 T**5,
                T = (jd - 2451545)/36252.

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


def get_asteroid_state(target, mjd, _jpl_eph, _epoch='J2000', _station=None):
    """ Compute obs parameters for a given t

    :param target: Kepler class object
    :param mjd: epoch in TDB/mjd (t.tdb.mjd, t - astropy.Time object, UTC)
    :param _jpl_eph: DE eph from pypride
    :param _epoch: 'J2000' (default), 'Date', or float (jdate like 2015.0)
    :param _station: None or pypride station object, if 'topocentric' ra/dec's are desired

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

    asteroid = Kepler(a, e, i, w, Node, M0, GSUN, t0, H, G)

    # jpl_eph - path to eph used by pypride
    radec, radec_dot, Vmag = asteroid.raDecVmag(mjd, jpl_eph=_jpl_eph, epoch=_epoch, station=_station, output_Vmag=True)

    return radec, radec_dot, Vmag


def asteroid_database_update(_f_database, _url='http://ssd.jpl.nasa.gov/dat/ELEMENTS.NUMBR', n=1.0):
    """
        Fetch an asteroid database update

        JPL: http://ssd.jpl.nasa.gov/dat/ELEMENTS.NUMBR
        MPC: http://www.minorplanetcenter.net/iau/MPCORB/ + [MPCORB.DAT, PHA.txt, NEA.txt, ...]
    """
    do_update = False
    if os.path.isfile(_f_database):
        age = datetime.datetime.now() - \
              datetime.datetime.utcfromtimestamp(os.path.getmtime(_f_database))
        if age.days > n:
            do_update = True
            print('MinorBody database: {:s} is out of date, updating...'.format(_f_database))
    else:
        do_update = True
        print('Database file: {:s} is missing, fetching...'.format(_f_database))
    # if the file is older than n days:
    if do_update:
        try:
            response = urllib2.urlopen(_url)
            with open(_f_database, 'w') as f:
                f.write(response.read())
        except Exception as err:
            print(str(err))
            pass


def asteroid_data_load(_f_database, asteroid_name):
    """ Load data from JPL database

    :param _f_database:
    :param asteroid_name:
    :return:
    """

    # update database if necessary:
    asteroid_database_update(_f_database)

    asteroid_number = int(asteroid_name.split('_')[0])

    l = linecache.getline(_f_database, asteroid_number+2)

    dt = np.dtype([('num', '<i8'), ('name', '|S21'),
                   ('epoch', '<i8'), ('a', '<f8'),
                   ('e', '<f8'), ('i', '<f8'),
                   ('w', '<f8'), ('Node', '<f8'),
                   ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
    return np.array([((int(l[0:6]),) + (l[6:25].strip(),) +
                               tuple(map(float, l[25:].split()[:-2])))], dtype=dt)


def unpack_epoch(epoch_str):
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


def asteroid_database_load(_f_database, _provider='mpc'):
    """
        Load MPC database
        :param _provider: 'mpc' or 'jpl'
    """

    # update database if necessary:
    if _provider == 'jpl':
        _url = 'http://ssd.jpl.nasa.gov/dat/ELEMENTS.NUMBR'
    elif _provider == 'mpc':
        _url = 'http://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT'
    else:
        raise Exception('ephemeris provider not recognized')

    asteroid_database_update(_f_database, _url=_url)

    with open(_f_database, 'r') as f:
        database = f.readlines()

    start = [i for i, l in enumerate(database[:300]) if l[0:2] == '--']
    if len(start) > 0:
        database = database[start[0] + 1:]
    # remove empty lines:
    database = [l for l in database if len(l) > 5]

    if _provider == 'jpl':
        dt = np.dtype([('name', '|S21'),
                       ('epoch', '<i8'), ('a', '<f8'),
                       ('e', '<f8'), ('i', '<f8'),
                       ('w', '<f8'), ('Node', '<f8'),
                       ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
        return np.array([((l[0:6] + '_' + l[6:25].strip(),) +
                          tuple(map(float, l[25:].split()[:-2]))) for l in database[2:]],
                        dtype=dt)

    elif _provider == 'mpc':
        dt = np.dtype([('designation', '|S21'), ('H', '<f8'), ('G', '<f8'),
                       ('epoch', '<f8'), ('M0', '<f8'), ('w', '<f8'),
                       ('Node', '<f8'), ('i', '<f8'), ('e', '<f8'),
                       ('n', '<f8'), ('a', '<f8'), ('U', '|S21'),
                       ('n_obs', '<f8'), ('n_opps', '<f8'), ('arc', '|S21'),
                       ('rms', '|S21'), ('name', '|S21'), ('last_obs', '|S21')
                      ])

        return np.array([(str(entry[:7]).strip(),) + (float(entry[8:13]) if len(entry[8:13].strip()) > 0 else 20.0,)
                           + (float(entry[14:19]) if len(entry[14:19].strip()) > 0 else 0.15,)
                           + (unpack_epoch(str(entry[20:25])),) + (float(entry[26:35]),) + (float(entry[37:46]),)
                           + (float(entry[48:57]),) + (float(entry[59:68]),) + (float(entry[70:79]),)
                           + (float(entry[80:91]) if len(entry[80:91].strip()) > 0 else 0,)
                           + (float(entry[92:103]) if len(entry[92:103].strip()) > 0 else 0,) + (str(entry[105:106]),)
                           + (int(entry[117:122]) if len(entry[117:122].strip()) > 0 else 0,)
                           + (int(entry[123:126]) if len(entry[123:126].strip()) > 0 else 0,) + (str(entry[127:136]).strip(),)
                           + (str(entry[137:141]),) + (str(entry[166:194]).strip().replace(' ', '_'),)
                           + (str(entry[194:202]).strip(),)
                         for entry in database], dtype=dt)


def asteroid_data_load_2(_f_database, asteroid_name, _provider='mpc'):
    """ Load data from JPL database

    :param _f_database:
    :param asteroid_name:
    :param _provider: 'mpc' or 'jpl'
    :return:
    """
    _database = asteroid_database_load(_f_database, _provider=_provider)
    print('lala')
    print(_database)

    ind = _database['name'] == asteroid_name

    return _database[ind]


def asteroid_data_load_3(_f_database, asteroid_name, _provider='mpc'):
    """ Load asteroid data from database

    :param _f_database:
    :param asteroid_name:
    :return:
    """
    name = ' '.join(asteroid_name.split('_'))

    try:

        with open(_f_database) as f:
            lines = f.readlines()
            entry = [line for line in lines if line.find(name) != -1][0]

        if _provider == 'jpl':
            dt = np.dtype([('name', '|S21'),
                           ('epoch', '<i8'), ('a', '<f8'),
                           ('e', '<f8'), ('i', '<f8'),
                           ('w', '<f8'), ('Node', '<f8'),
                           ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
            return np.array([((entry[0:6] + '_' + entry[6:25].strip(),) +
                              tuple(map(float, entry[25:].split()[:-2])))],
                            dtype=dt)

        elif _provider == 'mpc':
            dt = np.dtype([('designation', '|S21'), ('H', '<f8'), ('G', '<f8'),
                           ('epoch', '<f8'), ('M0', '<f8'), ('w', '<f8'),
                           ('Node', '<f8'), ('i', '<f8'), ('e', '<f8'),
                           ('n', '<f8'), ('a', '<f8'), ('U', '|S21'),
                           ('n_obs', '<f8'), ('n_opps', '<f8'), ('arc', '|S21'),
                           ('rms', '|S21'), ('name', '|S21'), ('last_obs', '|S21')
                          ])
            return np.array([(str(entry[:7]).strip(),) + (float(entry[8:13]) if len(entry[8:13].strip()) > 0 else 20.0,)
                               + (float(entry[14:19]) if len(entry[14:19].strip()) > 0 else 0.15,)
                               + (unpack_epoch(str(entry[20:25])),) + (float(entry[26:35]),) + (float(entry[37:46]),)
                               + (float(entry[48:57]),) + (float(entry[59:68]),) + (float(entry[70:79]),)
                               + (float(entry[80:91]) if len(entry[80:91].strip()) > 0 else 0,)
                               + (float(entry[92:103]) if len(entry[92:103].strip()) > 0 else 0,)
                               + (str(entry[105:106]),)
                               + (int(entry[117:122]) if len(entry[117:122].strip()) > 0 else 0,)
                               + (int(entry[123:126]) if len(entry[123:126].strip()) > 0 else 0,)
                               + (str(entry[127:136]).strip(),)
                               + (str(entry[137:141]),) + (str(entry[166:194]).strip().replace(' ', '_'),)
                               + (str(entry[194:202]).strip(),)], dtype=dt)
        else:
            raise Exception('ephemeris provider not recognized')

    except Exception as _e:
        print(_e)
        return None


def get_state_asteroid(_asteroid, _t, _jpl_eph):
    """
    :param _asteroid:
    :param _t:
    :param _jpl_eph:
    :return:
     current J2000 ra/dec of a moving object
     current J2000 ra/dec rates of a moving object
     if mag <= self.m_lim
    """
    radec, radec_dot, vmag = get_asteroid_state(_asteroid, _t, _jpl_eph)

    # reformat
    ra = '{:02.0f}:{:02.0f}:{:02.3f}'.format(*hms(radec[0]))
    dec = dms(radec[1])
    # fix zero padding for dec
    if dec[0] >= 0:
        dec = '{:02.0f}:{:02.0f}:{:02.3f}'.format(dec[0], abs(dec[1]), abs(dec[2]))
    else:
        dec = '{:03.0f}:{:02.0f}:{:02.3f}'.format(dec[0], abs(dec[1]), abs(dec[2]))
    ''' !!! NOTE: ra rate must be with a minus sign !!! '''
    ra_rate = '{:.5f}'.format(-radec_dot[0])
    dec_rate = '{:.5f}'.format(radec_dot[1])

    return ra, dec, ra_rate, dec_rate, vmag


def get_state_asteroid_astropy(_asteroid, _t, _jpl_eph, _epoch='J2000', _station=None):
    """
    :param _asteroid:
    :param _t:
    :param _jpl_eph:
    :param _epoch: 'J2000' (default), 'Date' or jdate
    :param _station: None (default) or pypride station object if 'topocentric' ra/dec's are desired

    :return:
     current ra/dec of a moving object at epoch
     current ra/dec rates of a moving object at epoch
     if mag <= self.m_lim
    """
    radec, radec_dot, vmag = get_asteroid_state(_asteroid, _t, _jpl_eph, _epoch, _station)

    # reformat
    ra = '{:02.0f}h{:02.0f}m{:02.3f}s'.format(*hms(radec[0]))
    dec = dms(radec[1])
    # fix zero padding for dec
    if dec[0] >= 0:
        dec = '{:02.0f}d{:02.0f}m{:02.3f}s'.format(dec[0], abs(dec[1]), abs(dec[2]))
    else:
        dec = '{:03.0f}d{:02.0f}m{:02.3f}s'.format(dec[0], abs(dec[1]), abs(dec[2]))
    ''' !!! NOTE: ra rate must be with a minus sign !!! '''
    ra_rate = '{:.5f}'.format(-radec_dot[0])
    dec_rate = '{:.5f}'.format(radec_dot[1])

    return ra, dec, ra_rate, dec_rate, vmag


def is_planet_or_moon(_name):
    """

    :param _name: body name
    :return:
    """
    planets = ('mercury', 'venus', 'earth', 'mars', 'jupiter',
               'saturn', 'uranus', 'neptune')
    moons = ('moon',
             'deimos', 'phobos',
             'europa', 'io', 'ganymede', 'callisto',
             'titan', 'enceladus', 'dione', 'hyperion', 'iapetus', 'mimas', 'rhea', 'tethys',
             'miranda', 'ariel', 'umbriel', 'oberon', 'titania')

    if _name.lower() in planets or 'pluto' in _name.lower() or _name.lower() in moons:
        return True
    else:
        return False


def get_state_planet_or_moon(body, t):
    """
        Get observational parameters for a SS target
    """
    # t = Time.now().utc.datetime

    b = None  # I don't like red
    exec('b = ephem.{:s}()'.format(body.title()))
    b.compute(ephem.Date(t), epoch='2000')

    # fix zero padding
    ra = str(b.a_ra)
    if ra.index(':') == 1:
        ra = '0' + ra
    dec = str(b.a_dec)
    if dec.index(':') == 1:
        dec = '0' + dec
    elif dec.index(':') == 2 and dec[0] == '-':
        dec = '-0' + dec[1:]

    # compute rates in arcsec/s:
    dt = datetime.timedelta(seconds=1)
    b.compute(ephem.Date(t + dt), epoch='2000')
    ra_p1 = b.a_ra*180.0/np.pi*3600.0
    dec_p1 = b.a_dec*180.0/np.pi*3600.0
    b.compute(ephem.Date(t - dt), epoch='2000')
    ra_m1 = b.a_ra*180.0/np.pi*3600.0
    dec_m1 = b.a_dec*180.0/np.pi*3600.0

    ra_rate = (ra_p1 - ra_m1)/2.0
    dec_rate = (dec_p1 - dec_m1)/2.0

    return ra, dec, ra_rate, dec_rate


def sso_state(_name, _time, _path_to_database, _path_to_jpl_eph, _epoch, _station=None):
    """

    :param _name:
    :param _time:
    :param _path_to_database:
    :param _path_to_jpl_eph:
    :param _epoch:
    :param _station:

    :return: asteroid state and vmag
    """
    if not is_planet_or_moon(_name):
        ''' asteroids '''
        try:
            asteroid = asteroid_data_load(_f_database=_path_to_database, asteroid_name=_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return np.zeros(5)

        try:
            ra, dec, ra_rate, dec_rate, vmag = get_state_asteroid_astropy(_asteroid=asteroid, _t=_time.tdb.mjd,
                                                                          _jpl_eph=_path_to_jpl_eph, _epoch=_epoch,
                                                                          _station=_station)
            # print(0, ra, dec, ra_rate, dec_rate, vmag)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return np.zeros(5)

    else:
        ''' planets and moons '''
        try:
            ra, dec, ra_rate, dec_rate = get_state_planet_or_moon(body=_name, t=_time.utc.datetime)
            vmag = 0
            # print(0, ra, dec, '{:.5f}'.format(ra_rate), '{:.5f}'.format(dec_rate))
        except Exception as e:
            print(e)
            traceback.print_exc()
            return np.zeros(5)

    return ra, dec, ra_rate, dec_rate, vmag


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
    eop_update(inp['cat_eop'], 3)

    # create parser
    parser = ArgumentParser(prog='sso_state.py',
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            description='Get state of a Solar system object.')
    # optional arguments
    # parser.add_argument('-p', '--parallel', action='store_true',
    #                     help='run computation in parallel mode')
    # positional arguments
    parser.add_argument('name', type=str, help='object name')
    parser.add_argument('ra_apr', type=str, help='object RA for middle of night')
    parser.add_argument('dec_apr', type=str, help='object Dec for middle of night')
    parser.add_argument('ra_rate_apr', type=str, help='object RA rate for middle of night')
    parser.add_argument('dec_rate_apr', type=str, help='object Dec rate for middle of night')
    parser.add_argument('time', type=str, help='UTC time for calculation (YYYYmmdd_HHMMSS[.ffffff])')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()

    name = args.name

    # a priori coordinates
    ra_apr = args.ra_apr
    dec_apr = args.dec_apr
    ra_rate_apr = args.ra_rate_apr
    dec_rate_apr = args.dec_rate_apr

    # get UTC time (astropy.time.Time object)
    if '.' in args.time:
        time = Time(str(datetime.datetime.strptime(args.time, '%Y%m%d_%H%M%S.%f')), format='iso', scale='utc')
    else:
        time = Time(str(datetime.datetime.strptime(args.time, '%Y%m%d_%H%M%S')), format='iso', scale='utc')

    if not is_planet_or_moon(name):
        ''' asteroids '''
        # asteroid database:
        path_to_database = config.get('Path', 'asteroid_database_path')
        # JPL:
        # f_database = os.path.join(path_to_database, 'ELEMENTS.NUMBR')
        # MPC:
        # f_database = os.path.join(path_to_database, 'MPCORB.dat')
        f_database = os.path.join(path_to_database, 'DAILY.DAT')

        try:
            # asteroid = asteroid_data_load(_f_database=f_database, asteroid_name=name)
            asteroid = asteroid_data_load_3(_f_database=f_database, asteroid_name=name, _provider='mpc')
            # asteroid = asteroid_data_load_3(_f_database=f_database, asteroid_name=name, _provider='jpl')
            # asteroid = asteroid_data_load_2(_f_database=f_database, asteroid_name=name, _provider='mpc')
            # print(asteroid)
        except Exception:
            # print error code 2 (failed to load asteroid database) and exit
            print(2, ra_apr, dec_apr, ra_rate_apr, dec_rate_apr)
            raise SystemExit

        try:
            ra, dec, ra_rate, dec_rate, _ = get_state_asteroid(_asteroid=asteroid, _t=time.tdb.mjd,
                                                               _jpl_eph=inp['jpl_eph'])
            print(0, ra, dec, ra_rate, dec_rate)
        except Exception:
            # print error code 4 (calculation failed) and exit
            print(traceback.print_exc())
            print(3, ra_apr, dec_apr, ra_rate_apr, dec_rate_apr)
            raise SystemExit

    else:
        ''' planets and moons '''
        try:
            ra, dec, ra_rate, dec_rate = get_state_planet_or_moon(body=name, t=time.utc.datetime)
            print(0, ra, dec, '{:.5f}'.format(ra_rate), '{:.5f}'.format(dec_rate))
        except Exception:
            # print error code 4 (calculation failed) and exit
            print(3, ra_apr, dec_apr, ra_rate_apr, dec_rate_apr)
            raise SystemExit
