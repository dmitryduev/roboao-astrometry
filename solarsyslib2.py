from __future__ import print_function
import numpy as np
import os
import datetime
import urllib2
from pypride.classes import inp_set
from pypride.vintflib import pleph
from pypride.vintlib import sph2cart, cart2sph, iau_PNM00A
from pypride.vintlib import eop_update, mjuliandate
from astropy.time import Time
from numba import jit
from copy import deepcopy
import traceback


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
        raise NotImplementedError


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
        raise NotImplementedError


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
    # run tests:
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
    # pass
