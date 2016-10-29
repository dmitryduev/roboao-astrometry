# coding=utf-8
"""
The Asteroids Galaxy Tour

@autor: Dr Dmitry A. Duev

this is how to profile this lib:

import pstats
stats = pstats.Stats('/Users/dmitryduev/Library/Caches/PyCharm50/snapshots/solsy.pstat')
stats.sort_stats('total')
stats.print_stats(20)

"""

from __future__ import print_function

import datetime
import pytz
import os
import urllib2
from copy import deepcopy
from time import time as _time

import multiprocessing
import numpy as np
# from astroplan import Observer, FixedTarget
from astroplan import Observer as Observer_astroplan
from astroplan import FixedTarget
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
from astropy.time import Time
from astropy.coordinates import AltAz
from astroquery.vizier import Vizier
from pypride.classes import inp_set
from pypride.vintflib import pleph
from pypride.vintlib import factorise, aber_source, R_123, aber_source
from pypride.vintlib import taitime, eop_iers, t_eph, ter2cel, \
    load_cats, sph2cart, cart2sph, iau_PNM00A
from pypride.vintlib import eop_update

# from astroplan import FixedTarget
from astroplan import observability_table  # , is_observable, is_always_observable
# from astroplan.plots import plot_sky
from astroplan import time_grid_from_range
from astroplan import AtNightConstraint, AltitudeConstraint

import ephem

import xml.etree.ElementTree as ET
from xml.dom import minidom
from dicttoxml import dicttoxml
from collections import OrderedDict
import requests

import matplotlib.pyplot as plt
import seaborn as sns

from time import time as _time
from numba import jit

# from astroplan import download_IERS_A
# download_IERS_A()

sns.set_style('whitegrid')
plt.close('all')
sns.set_context('talk')

Vizier.ROW_LIMIT = -1
Vizier.TIMEOUT = 3600


def _target_is_vector(target):
    if hasattr(target, '__iter__'):
        return True
    else:
        return False


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


def asteroid_multiples_numbers():
    _f_lines = '''  (134340) Pluto, Charon, Nix, Hydra, Kerberos, and Styx
                    (22) Kalliope and Linus
                    (41) Daphne and S/2008 (41) 1
                    (45) Eugenia, Petit-Prince, and S/2004 (45) 1
                    (87) Sylvia, Romulus, and Remus
                    (90) Antiope and S/2000 (90) 1
                    (93) Minerva, Aegis, and Gorgoneion
                    (107) Camilla and S/2001 (107) 1
                    (121) Hermione and S/2002 (121) 1
                    (130) Elektra, S/2003 (130) 1, and S/2014 (130) 1
                    (216) Kleopatra, Alexhelios, and Cleoselene
                    (243) Ida and Dactyl
                    (283) Emma and S/2003 (283) 1
                    (317) Roxane and S/2009 (317) 1
                    (379) Huenna and S/2003 (379) 1
                    (617) Patroclus and Menoetius
                    (624) Hektor and S/2006 (624) 1
                    (702) Alauda and Pichi unem
                    (762) Pulcova
                    (809) Lundia
                    (854) Frostia
                    (939) Isberga
                    (1052) Belgica
                    (1089) Tama
                    (1139) Atami
                    (1313) Berna
                    (1333) Cevenola
                    (1338) Duponta
                    (1453) Fennia
                    (1509) Esclangona and S/2003 (1509) 1
                    (1717) Arlon
                    (1727) Mette
                    (1830) Pogson
                    (1862) Apollo
                    (1866) Sisyphus
                    (2006) Polonskaya
                    (2044) Wirt
                    (2047) Smetana
                    (2121) Sevastopol
                    (2131) Mayall
                    (2343) Siding Spring
                    (2449) Kenos
                    (2478) Tokai
                    (2486) Metsahovi
                    (2577) Litva, S/2012 (2577) 1, and second satellite
                    (2623) Zech
                    (2691) Sersic
                    (2754) Efimov
                    (2815) Soma
                    (3034) Climenhaga
                    (3073) Kursk
                    (3169) Ostro
                    (3309) Brorfelde
                    (3433) Fehrenbach
                    (3671) Dionysus and S/1997 (3671) 1
                    (3673) Levy
                    (3703) Volkonskaya
                    (3749) Balam, S/2002 (3749) 1, and third component
                    (3782) Celle
                    (3841) Dicicco
                    (3868) Mendoza
                    (3873) Roddy
                    (3905) Doppler
                    (3951) Zichichi
                    (3982) Kastel
                    (4029) Bridges
                    (4272) Entsuji
                    (4383) Suruga
                    (4440) Tchantches
                    (4492) Debussy
                    (4514) Vilen
                    (4541) Mizuno
                    (4607) Seilandfarm
                    (4666) Dietz
                    (4674) Pauling and S/2004 (4674) 1
                    (4765) Wasserburg
                    (4786) Tatianina
                    (4868) Knushevia
                    (4951) Iwamoto
                    (5143) Heracles
                    (5261) Eureka
                    (5381) Sekhmet
                    (5407) 1992 AX
                    (5425) Vojtech
                    (5426) Sharp
                    (5474) Gingasen
                    (5477) Holmes
                    (5481) Kiuchi
                    (5646) 1990 TR
                    (5899) Jedicke
                    (5905) Johnson
                    (6084) Bascom
                    (6244) Okamoto
                    (6265) 1985 TW3
                    (6369) 1983 UC
                    (6615) Plutarchos
                    (6708) Bobbievaile
                    (7088) Ishtar
                    (7187) Isobe
                    (7225) Huntress
                    (7369) Gavrilin
                    (7888) 1993 UC
                    (7958) Leakey
                    (8026) Johnmckay
                    (8116) Jeanperrin
                    (8306) Shoko
                    (8373) Stephengould
                    (8474) Rettig
                    (9069) Hovland
                    (9260) Edwardolson
                    (9617) Grahamchapman
                    (9783) Tensho-kan
                    (10123) Fideoja
                    (10199) Chariklo and ring system
                    (10208) Germanicus
                    (11217) 1999 JC4
                    (11264) Claudiomaccone
                    (13123) Tyson
                    (15268) Wendelinefroger
                    (15430) 1998 UR31
                    (15700) 1987 QD
                    (15822) 1994 TV15
                    (16525) Shumarinaiko
                    (16635) 1993 QO
                    (17246) 2000 GL74 and S/2004 (17246) 1
                    (17260) 2000 JQ58
                    (17365) 1978 VF11
                    (18890) 2000 EV26
                    (20325) 1998 HO27
                    (21436) Chaoyichi
                    (22899) 1999 TO14 and S/2003 (22899) 1
                    (26074) Carlwirtz
                    (26308) 1998 SM165 and S/2001 (26308) 1
                    (26416) 1999 XM84
                    (26471) 2000 AS152
                    (27568) 2000 PT6
                    (29314) Eurydamas
                    (31345) 1998 PG
                    (31450) 1999 CU9
                    (32008) 2000 HM53
                    (32039) 2000 JO23
                    (34706) 2001 OP83
                    (35107) 1991 VH and S/2008 (35107) 1
                    (38628) Huya
                    (42355) Typhon and Echidna
                    (43008) 1999 UD31
                    (44620) 1999 RS43
                    (47171) 1999 TC36, S/2001 (47171) 1, and third component
                    (46829) 1998 OS14
                    (48639) 1995 TL8
                    (50000) Quaoar and Weywot
                    (51356) 2000 RY76
                    (52316) 1992 BD
                    (53432) 1999 UT55
                    (55637) 2002 UX25
                    (58534) Logos and Zoe
                    (60458) 2000 CM114
                    (60621) 2000 FE8
                    (65489) Ceto and Phorcys
                    (65803) Didymos
                    (66063) 1998 RO1
                    (66391) 1999 KW4
                    (66652) Borasisi and Pabu
                    (69230) Hermes
                    (69406) 1995 SX48
                    (76818) 2000 RG79
                    (79360) Sila-Nunam
                    (79472) 1998 AX4
                    (80218) 1999 VO123
                    (80806) 2000 CM105
                    (82075) 2000 YW134
                    (82157) 2001 FM185
                    (85938) 1999 DJ4
                    (88611) Teharonhiawako and Sawiskera
                    (88710) 2001 SL9
                    (90482) Orcus and Vanth
                    (99913) 1997 CZ5
                    (114319) 2002 XD58
                    (119067) 2001 KP76
                    (119979) 2002 WC19
                    (120347) Salacia and Actaea
                    (123509) 2000 WK183
                    (134860) 2000 OJ67
                    (136108) Haumea, Hi'iaka, and Namaka
                    (136199) Eris and Dysnomia
                    (136617) 1994 CC
                    (136993) 1998 ST49
                    (137170) 1999 HF1
                    (138095) 2000 DK79
                    (139775) 2001 QG298
                    (148780) Altjira
                    (153591) 2001 SN263
                    (153958) 2002 AM31
                    (160091) 2000 OL67
                    (160256) 2002 PD149
                    (162000) 1990 OS
                    (162483) 2000 PJ5
                    (164121) 2003 YT1
                    (174567) Varda and Ilmare
                    (175706) 1996 FG3
                    (182933) 2002 GZ31
                    (185851) 2000 DP107 and S/2000 (2000 DP107) 1
                    (190208) 2006 AQ
                    (208996) 2003 AZ84
                    (218144) 2002 RL66
                    (229762) 2007 UK126
                    (275809) 2001 QY297
                    (276049) 2002 CE26
                    (285263) 1998 QE2
                    (303712) 2005 PR21
                    (311066) 2004 DC
                    (341520) Mors-Somnus
                    (348400) 2005 JF21
                    (357439) 2004 BL86
                    (363027) 1998 ST27
                    (363067) 2000 CO101
                    (363599) 2004 FG11
                    (364171) 2006 JZ81
                    (374851) 2006 VV2
                    (385186) 1994 AW1
                    (385446) Manwe and Thorondor
                    (399307) 1991 RJ2
                    (399774) 2005 NB7
                    (410777) 2009 FD
                    (450894) 2008 BT18
                    (452561) 2005 AB '''

    multiples_numbers = np.array([int(_l.split()[0].replace('(', '').replace(')', ''))
                                for _l in _f_lines.split('\n')])
    return multiples_numbers


def is_multiple_asteroid(name):
    """

    :param name:
    :param _f_base:
    :return:
    """
    try:
        number = int(name.split('_')[0])
    except Exception as err:
        print(str(err))
        print('could not get the asteroid number.')
        return False

    if number in asteroid_multiples_numbers():
        return True
    else:
        return False


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
        #        eps = 23.43929111*np.pi/180.0
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
            Compute (geocentric) precession/nutation matrix for epoch
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
        _, _, eops = load_cats(inp, 'DUMMY', 'S', ['GEOCENTR'], tstamp)

        ''' interpolate eops to tstamp '''
        UT1, eop_int = eop_iers(mjd, UTC, eops)

        ''' compute coordinate time fraction of CT day at GC '''
        CT, dTAIdCT = t_eph(JD, UT1, TT, 0.0, 0.0, 0.0)

        ''' rotation matrix IERS '''
        r2000 = ter2cel(tstamp, eop_int, dTAIdCT, 'iau2000')
        #    print(r2000[:,:,0])

        return r2000

    @jit
    def raDecVmag(self, mjd, jpl_eph, epoch='J2000'):
        """ Calculate ra/dec's from equatorial state
            Then compute asteroid's expected visual magnitude

        :param mjd: MJD epoch in decimal days
        :param jpl_eph: target's heliocentric equatorial
        :param epoch: RA/Dec epoch. 'J2000' or 'Date'

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

        # quick and dirty (but accurate enough for pointing, I hope)
        # LT-correction:
        C = 299792458.0
        lt = np.linalg.norm(earth[:, 0] - (sun[:, 0] + state[:, 0])) / C
        #        print(lt)

        # recompute:
        # Sun:
        rrd = pleph(jd - lt / 86400.0, 11, 12, jpl_eph)
        sun = np.reshape(np.asarray(rrd), (3, 2), 'F') * 1e3
        # target state:
        state = self.ecliptic_to_equatorial(self.to_cart(mjd - lt / 86400.0))

        r = (sun[:, 0] + state[:, 0]) - earth[:, 0]
        # RA/Dec J2000:
        ra = np.arctan2(r[1], r[0])  # right ascension
        dec = np.arctan(r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2))  # declination
        if ra < 0:
            ra += 2.0 * np.pi

        # go for time derivatives:
        v = (sun[:, 1] + state[:, 1]) - earth[:, 1]
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
        if epoch != 'J2000' or epoch == 'Date':
            xyz2000 = sph2cart(np.array([1.0, dec, ra]))
            rDate = iau_PNM00A(jd, 0.0)
            xyzDate = np.dot(rDate, xyz2000)
            dec, ra = cart2sph(xyzDate)[1:]
            if ra < 0:
                ra += 2.0 * np.pi

        ''' go for Vmag based on H-G model '''
        if self.H is None or self.G is None:
            print('Can\'t compute Vmag - no H-G model data provided.')
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

            AU_DE421 = 1.49597870699626200e+11  # m

            Vmag = self.H - 2.5 * np.log10((1.0 - self.G) * phi1 + self.G * phi2) + \
                   5.0 * np.log10(EA_norm * SA_norm / AU_DE421 ** 2)

        # returning SkyCoord is handy, but very expensive
        # return (SkyCoord(ra=ra, dec=dec, unit=(u.rad, u.rad), frame='icrs'),
        #         (ra_dot, dec_dot), Vmag)
        return [ra, dec], [ra_dot, dec_dot], Vmag


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
    meridian_transit = targlist.get_hour_angle_limit2(night, radec[0], radec[1], N=20)
    return [meridian_transit]


class TargetListPlanetsAndMoons(object):
    """
        Produce (nightly) target list for the Solar System weather project
    """

    def __init__(self, _f_inp, _observatory='kitt peak', _m_lim=16.5):
        # observatory object
        self.observatory = Observer.at_site(_observatory)

        # minimum object magnitude to be output
        self.m_lim = _m_lim

        # inp file for running pypride:
        inp = inp_set(_f_inp)
        self.inp = inp.get_section('all')

    def middle_of_night(self, day):
        """
            day - datetime.datetime object, 0h UTC of the coming day
        """
        #        day = datetime.datetime(2015,11,7) # for KP, in UTC it is always 'tomorrow'
        nextDay = day + datetime.timedelta(days=1)
        # print(nextDay)
        astrot = Time([str(day), str(nextDay)], format='iso', scale='utc')
        # when the night comes, heh?
        sunSet = self.observatory.sun_set_time(astrot[0])
        sunRise = self.observatory.sun_rise_time(astrot[1])

        night = Time([str(sunSet.datetime), str(sunRise.datetime)],
                     format='iso', scale='utc')

        # build time grid for the night to come
        time_grid = time_grid_from_range(night)
        middle_of_night = time_grid[len(time_grid) / 2]
        # print(middle_of_night.datetime)

        return night, middle_of_night

    def target_list_all(self, day):
        """
            Get observational parameters for a (masked) target list
            from self.database
        """
        # get middle of night:
        night, middle_of_night = self.middle_of_night(day)
        # mjd = middle_of_night.tdb.mjd  # in TDB!!
        t = middle_of_night.datetime
        # print(t)
        # go over planets/moons:
        bodies = ['venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto',
                  'phobos',
                  'europa', 'io', 'ganymede', 'callisto',
                  'titan', 'enceladus', 'iapetus', 'mimas']
        target_list = []

        b = None  # I don't like red
        for body in bodies:
            exec('b = ephem.{:s}()'.format(body.title()))

            b.compute(ephem.Date(t))
            if body == 'europa':
                b.mag = 5.29
            elif body == 'io':
                b.mag = 5.02
            elif body == 'ganymede':
                b.mag = 4.61
            elif body == 'callisto':
                b.mag = 5.65
            elif body == 'titan':
                b.mag = 8.5
            elif body == 'enceladus':
                b.mag = 11.7
            elif body == 'iapetus':
                b.mag = 11.0
            elif body == 'mimas':
                b.mag = 12.9
            elif body == 'phobos':
                b.mag = 11.3

            ra = b.a_ra
            dec = b.a_dec
            try:
                mag = b.mag
            except AttributeError:
                b.mag = 9.99
                mag = b.mag

            # compute rates in arcsec/s:
            sec = 1
            dt = datetime.timedelta(seconds=sec)
            b.compute(ephem.Date(t + dt))
            ra_p1 = b.a_ra*180.0/np.pi*3600.0
            dec_p1 = b.a_dec*180.0/np.pi*3600.0
            b.compute(ephem.Date(t - dt))
            ra_m1 = b.a_ra*180.0/np.pi*3600.0
            dec_m1 = b.a_dec*180.0/np.pi*3600.0

            ra_rate = (ra_p1 - ra_m1)/(2.0*sec)
            dec_rate = (dec_p1 - dec_m1)/(2.0*sec)

            # calculate meridian transit time to set hour angle limit.
            # no need to observe a planet when it's low if can wait until it's high.
            # get next transit after night start:
            meridian_transit_time = self.observatory.target_meridian_transit_time(night[0],
                                        FixedTarget(coord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad),
                                        name=body.title()), which='next')

            # will it happen during the night?
            meridian_transit = night[0] <= meridian_transit_time <= night[1]
            # print(night[0] <= meridian_transit_time <= night[1])
            # print(night[0], meridian_transit_time.iso, night[1], '\n')

            print(body, ra, dec, ra_rate, dec_rate, mag, meridian_transit)

            target_list.append([{'name': body.title()}, middle_of_night,
                                [ra, dec], [ra_rate, dec_rate], mag, meridian_transit])

        return np.array(target_list)

    def target_list_observable(self, target_list, day,
                               elv_lim=40, twilight='nautical', fraction=0.1):
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
        print('observability computation took: ', _time() - tic)

        # proceed with observable (for more than 5% of the night) targets only
        mask_observable = table['fraction of time observable'] > fraction

        target_list_observeable = target_list[mask_observable]
        print('total bright: ', len(target_list), 'observable: ', len(target_list_observeable))
        return target_list_observeable


class TargetListAsteroids(object):
    """
        Produce (nightly) target list for the asteroids project
    """

    def __init__(self, _f_database, _f_inp, _observatory='kitt peak', _m_lim=16.5,
                 date=None, timezone='America/Phoenix'):
        # update database
        self.asteroid_database_update(_f_database)
        # load it to self.database:
        self.database = self.asteroid_database_load(_f_database)
        # observatory object
        self.observatory = Observer.at_site(_observatory)

        # minimum object magnitude to be output
        self.m_lim = _m_lim

        # inp file for running pypride:
        inp = inp_set(_f_inp)
        self.inp = inp.get_section('all')

        # update pypride eops
        eop_update(self.inp['cat_eop'], 3)

        ''' load eops '''
        if date is None:
            now = datetime.datetime.now(pytz.timezone(timezone))
            date = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)
        _, _, self.eops = load_cats(self.inp, 'DUMMY', 'S', ['GEOCENTR'], date)

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

        t = np.linspace(0, 1, N)
        p = np.polyfit(t, altitudes, 6)
        dense = np.polyval(p, np.linspace(0, 1, 200))
        maxp = np.max(dense)
        root = np.argmax(dense)/200.0
        minus = np.polyval(p, maxp - 0.01)
        plus = np.polyval(p, maxp + 0.01)

        # print('pypride ', (time[0] + root * u.day).iso)
        return (time[0] + root*u.day < time[1]) and (np.max((minus, plus)) < maxp)

    def get_current_state(self, name):
        """
        :param name:
        :return:
         current J2000 ra/dec of a moving object
         current J2000 ra/dec rates of a moving object
         if mag <= self.m_lim
        """
        number = int(name.split('_')[0])
        asteroid = self.database[number-1]

        radec, radec_dot, _ = self.getObsParams(asteroid, Time.now().tdb.mjd)

        # reformat
        ra = '{:02.0f}:{:02.0f}:{:02.3f}'.format(*hms(radec[0]))
        dec = dms(radec[1])
        dec = '{:02.0f}:{:02.0f}:{:02.3f}'.format(dec[0], abs(dec[1]), abs(dec[2]))
        ''' !!! NOTE: ra rate must be with a minus sign for the TCS !!! '''
        ra_rate = '{:.5f}'.format(-radec_dot[0])
        dec_rate = '{:.5f}'.format(radec_dot[1])

        return ra, dec, ra_rate, dec_rate

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
            meridian_transiting_asteroids = np.array(result.get())
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
                    meridian_transit = self.get_hour_angle_limit2(night, radec[0], radec[1], N=20)
                    # print(_time() - ticcc, meridian_transit)
                    target_list.append([asteroid, middle_of_night,
                                        radec, radec_dot, Vmag, meridian_transit])
            target_list = np.array(target_list)
            print('serial computation took: {:.2f} s'.format(_time() - ttic))
        # print('Total targets brighter than 16.5', len(target_list))
        return target_list

    def target_list_observable(self, target_list, day,
                               elv_lim=40, twilight='nautical', fraction=0.1):
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

        asteroid = Kepler(a, e, i, w, Node, M0, GSUN, t0, H, G)

        # jpl_eph - path to eph used by pypride
        radec, radec_dot, Vmag = asteroid.raDecVmag(mjd, self.inp['jpl_eph'])
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

    @staticmethod
    def asteroid_database_update(_f_database, n=1):
        """
            Fetch a database update from JPL
        """
        do_update = False
        if os.path.isfile(_f_database):
            age = datetime.datetime.now() - \
                  datetime.datetime.utcfromtimestamp(os.path.getmtime(_f_database))
            if age.days > n:
                do_update = True
                print('Asteroid database: {:s} is out of date, updating...'.format(_f_database))
        else:
            do_update = True
            print('Database file: {:s} is missing, fetching...'.format(_f_database))
        # if the file is older than n days:
        if do_update:
            try:
                response = urllib2.urlopen('http://ssd.jpl.nasa.gov/dat/ELEMENTS.NUMBR')
                with open(_f_database, 'w') as f:
                    f.write(response.read())
            except Exception as err:
                print(str(err))
                pass

    @staticmethod
    def asteroid_database_load(_f_database):
        """
            Load JPL database
        """
        with open(_f_database, 'r') as f:
            database = f.readlines()

        dt = np.dtype([('num', '<i8'), ('name', '|S21'),
                       ('epoch', '<i8'), ('a', '<f8'),
                       ('e', '<f8'), ('i', '<f8'),
                       ('w', '<f8'), ('Node', '<f8'),
                       ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
        return np.array([((int(l[0:6]),) + (l[6:25].strip(),) +
                          tuple(map(float, l[25:].split()[:-2]))) for l in database[2:]],
                        dtype=dt)


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
                                                              ('filter_code', 'FILTER_LONGPASS_600'),
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
                                                              ('filter_code', 'FILTER_SLOAN_R'),
                                                              ('camera_mode', ''),
                                                              ('repeat_times', 1),
                                                              ('repeated', 0),
                                                              ('done', 0)]),
                                                 OrderedDict([('number', 3),
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
            #                print target_xml_path

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
                    xml['comment'] = 'modified {:s}'.format(str(datetime.datetime.now()))
                elif is_multiple_asteroid(name):
                    xml['comment'] = 'known multiple; modified {:s}'.format(str(datetime.datetime.now()))
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
                ind_num_obs = [i+start for i, v in enumerate(target_xml[start:stop])
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

        target_list_xml = ['Target_{:d}.xml'.format(i+1) for i in range(int(pnot))]

        for targ_num, target_xml in enumerate(target_list_xml):
            tree = ET.parse(os.path.join(self.path, target_xml))
            root = tree.getroot()

            targ = {}
            targ['Object'] = []
            for content in root:
                if content.tag != 'Object':
                    targ[content.tag] = content.text
                else:
                    obj = {}
                    obj['Observation'] = []
                    for data_obj in content:
                        if data_obj.tag != 'Observation':
                            obj[data_obj.tag] = data_obj.text
                        else:
                            obs = {}
                            for data_obs in data_obj:
                                obs[data_obs.tag] = data_obs.text
                            obj['Observation'].append(obs)
                    targ['Object'].append(obj)

            try:
                # TODO: the queue software does not like spaces even in comments
                t_xml = datetime.datetime.strptime(' '.join(targ['comment'].split()[-2:]),
                                                   '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # why not?
                t_xml = datetime.datetime.now() - datetime.timedelta(days=1)
            # updated > 2 days ago?
            if (datetime.datetime.now() - t_xml).total_seconds() > 86400*2 \
                    and targ['done'] == '0' \
                    and targ['Object'][0]['Observation'][0]['repeated'] == '0':
                # print(targ['comment'], targ['done'])
                target_nums_to_remove.append(targ_num+1)

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


class GuideStar(object):
    """
        Keep the guide star netjes in one place
    """

    def __init__(self, ra=None, dec=None, separation=None, mag=None):
        """
        :param ra: RA in degrees
        :param dec: DEC in degrees
        :param separation: in arcseconds
        :param mag: magnitude
        """

        if ra is not None and dec is not None:
            self.crd = SkyCoord(ra=ra, dec=dec,
                                unit=(u.deg, u.deg), frame='icrs')
        else:
            self.crd = None

        self.separation = separation

        if mag is not None:
            self.mag = mag
        else:
            # placeholder:
            # F-red, j-Bj(blue), V-green, N-0.8um, U, B-blue
            self.mag = {'jmag': 50, 'Bmag': 50, 'Vmag': 50, 'Fmag': 50,
                        'Nmag': 50, 'Umag': 50}

    def __str__(self):
        if self.crd is not None:
            st = '{:02.0f}h{:02.0f}m{:06.3f}s '. \
                     format(self.crd.ra.hms[0], self.crd.ra.hms[1], self.crd.ra.hms[2]) + \
                 '{:+03.0f}d{:02.0f}\'{:06.3f}\" {:6.3f} '. \
                     format(self.crd.dec.dms[0],
                            abs(self.crd.dec.dms[1]), abs(self.crd.dec.dms[2]),
                            self.separation)
        else:
            st = 'Empty GuideStar object'
        for band in self.mag:
            if self.mag[band] != 50:
                st += '{:s}: {:.3f} '.format(band, self.mag[band])
        return st

    def setCrd(self, ra, dec):
        self.crd = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    def setSeparation(self, separation):
        self.separation = separation

    def setMag(self, band, mag):
        self.mag[band] = mag


class BrightestInBand(object):
    """

    """

    def __init__(self):

        self.bands = {'jmag': GuideStar(), 'Bmag': GuideStar(),
                      'Vmag': GuideStar(), 'Fmag': GuideStar(),
                      'Nmag': GuideStar(), 'Umag': GuideStar()}

    def __str__(self):
        st = ''
        for band in self.bands.keys():
            if self.bands[band].mag[band] != 50:
                st += '#{:s}_band: '.format(band[0])
                st += '{:02.0f}h{:02.0f}m{:06.3f}s '. \
                          format(self.bands[band].crd.ra.hms[0],
                                 self.bands[band].crd.ra.hms[1],
                                 self.bands[band].crd.ra.hms[2]) + \
                      '{:+03.0f}d{:02.0f}\'{:06.3f}\" {:6.3f} '. \
                          format(self.bands[band].crd.dec.dms[0],
                                 abs(self.bands[band].crd.dec.dms[1]),
                                 abs(self.bands[band].crd.dec.dms[2]),
                                 self.bands[band].separation)
                for band_gs in self.bands.keys():
                    if self.bands[band].mag[band_gs] != 50:
                        st += '{:s}: {:.3f} '.format(band_gs,
                                                     self.bands[band].mag[band_gs])
        return st


def get_gss(table):
    """ Get brightest star in band

    :param table:
    :return:
    """

    bgsib = BrightestInBand()
    for band in bgsib.bands.keys():
        try:
            mag = map(float, table[band])
            min_ind = np.argmin(mag)
            mag = mag[min_ind]
            if table[band].mask[min_ind] != True and \
                            mag < bgsib.bands[band].mag[band]:
                ra = float(table['_RAJ2000'][min_ind])
                dec = float(table['_DEJ2000'][min_ind])
                sep = float(table['_r'][min_ind])
                bgsib.bands[band].setCrd(ra, dec)
                bgsib.bands[band].setSeparation(sep)
                for band_gs in bgsib.bands.keys():
                    if table[band_gs].mask[min_ind] != True:
                        bgsib.bands[band].setMag(band_gs,
                                                 float(table[band_gs][min_ind]))
        except:
            continue

    if (bgsib.bands['Bmag'].mag['Bmag'] <= 16.5) or \
            (bgsib.bands['Vmag'].mag['Vmag'] <= 16.5) or \
            (bgsib.bands['Fmag'].mag['Fmag'] <= 16.5) or \
            (bgsib.bands['jmag'].mag['jmag'] <= 16.5) or \
            (bgsib.bands['Nmag'].mag['Nmag'] <= 14) or \
            (bgsib.bands['Umag'].mag['Umag'] <= 13):
        return bgsib
    else:
        return None


def get_guide_star(target_table, chunk=None, ang_sep='33s'):
    factors = factorise(len(target_table))
    # split in equal chunks
    chunk = factors[np.searchsorted(factors, chunk)] \
        if chunk is not None else chunk
    nChunks = len(target_table) / chunk

    # Look up guide stars in GSC2.3.2
    # read in the first chunk:
    tic = _time()
    guide = Vizier(catalog=u'I/305/out').query_region(target_table[:chunk],
                                                      radius=ang_sep)[0]
    print('Chunk #1 took {:f} s\n'.format(_time() - tic))
    for nC in range(1, nChunks):
        print('{:d} chunks of {:d} objects to go...'.format(nChunks - nC, chunk))
        tic = _time()
        tmp = Vizier(catalog=u'I/305/out'). \
            query_region(target_table[chunk * nC:chunk * (nC + 1)],
                         radius=ang_sep)[0]
        # fix numbering:
        tmp['_q'] += nC * chunk
        guide = vstack((guide, tmp))
        print('Chunk #{:d} took {:f} s\n'.format(nC + 1, _time() - tic))

    # filter out those with a bright enough star
    for it, target in enumerate(target_table):
        print('{:d} targets to go'.format(len(target_table) - it))
        mask_iq = guide['_q'] == it + 1

        gss = get_gss(guide[mask_iq])

        if gss is not None:
            print(it + 1, gss)

    return guide
