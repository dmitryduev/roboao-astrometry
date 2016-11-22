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
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from time import time as _time
import multiprocessing
import pytz
from numba import jit
from copy import deepcopy
import traceback


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


class TargetListAsteroids(object):
    """
        Produce (nightly) target list for the asteroids project
    """

    def __init__(self, _f_inp, database_source='mpc', _observatory='kitt peak', _m_lim=16.0,
                 date=None, timezone='America/Phoenix'):
        # init database
        if database_source == 'mpc':
            db = AsteroidDatabaseMPC()
        elif database_source == 'jpl':
            db = AsteroidDatabaseJPL()
        else:
            raise Exception('database source not understood')
        # load the database
        db.load()

        self.database = db.database
        # observatory object
        self.observatory = Observer.at_site(_observatory)

        # minimum object magnitude to be output
        self.m_lim = _m_lim

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

    def get_hour_angle_limit(self, night, ra, dec):
        # calculate meridian transit time to set hour angle limit.
        # no need to observe a planet when it's low if can wait until it's high.
        # get next transit after night start:
        meridian_transit_time = self.observatory.\
                                target_meridian_transit_time(night[0],
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
            target_list = [[database_masked[_it]] + [middle_of_night] + _t
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
                    target_list.append([asteroid, middle_of_night,
                                        radec, radec_dot, Vmag, meridian_transit, t_azel])
            target_list = np.array(target_list)
            print('serial computation took: {:.2f} s'.format(_time() - ttic))
        # print('Total targets brighter than 16.5', len(target_list))
        return target_list

    def target_list_observable(self, target_list, day,
                               elv_lim=40.0, twilight='nautical', fraction=0.1):
        """ Check whether targets are observable and return only those

        :param target_list:
        :param day:
        :param elv_lim:
        :param twilight:
        :param fraction:
        :return:
        """

        night, middle_of_night = self.middle_of_night(day)
        # set constraints (above elv_lim deg altitude, Sun altitude < -N deg [dep.on twilight])
        constraints = [AltitudeConstraint(elv_lim * u.deg, 90 * u.deg)]
        if twilight == 'nautical':
            constraints.append(AtNightConstraint.twilight_nautical())
        elif twilight == 'astronomical':
            constraints.append(AtNightConstraint.twilight_astronomical())
        elif twilight == 'civil':
            constraints.append(AtNightConstraint.twilight_civil())

        radec = np.array(list(target_list[:, 2]))
        # tic = _time()
        coords = SkyCoord(ra=radec[:, 0], dec=radec[:, 1],
                          unit=(u.rad, u.rad), frame='icrs')
        # print(_time() - tic)
        tic = _time()
        table = observability_table(constraints, self.observatory, coords,
                                    time_range=night)
        print('observability computation took: {:.2f} s'.format(_time() - tic))
        print(table)

        # proceed with observable (for more than 5% of the night) targets only
        mask_observable = table['fraction of time observable'] > fraction

        target_list_observeable = target_list[mask_observable]
        print('total bright asteroids: ', len(target_list),
              'observable: ', len(target_list_observeable))
        return target_list_observeable

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
    m_lim = 18
    # elevation cut-off [deg]:
    elv_lim = float(config.get('Asteroids', 'elv_lim'))

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

    tl = TargetListAsteroids(f_inp, database_source='mpc', _observatory=observatory, _m_lim=m_lim, date=today)
    mask = None
    targets = tl.target_list_observable(tl.target_list_all(today, mask, parallel=True), today,
                                        elv_lim=elv_lim, twilight=twilight, fraction=fraction)

    print(targets)
