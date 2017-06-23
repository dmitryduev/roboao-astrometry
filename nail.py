from __future__ import print_function

import argparse
from copy import deepcopy

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


np.set_printoptions(precision=16)


def xy2uv(p, x_raw, y_raw, xy_center=(0, 0), drizzled=False):
    """
        Convert (x, y) pixel position on CCD to (u, v) position on tangential plane (in degrees)
    :param p:
    :param x_raw:
    :param y_raw:
    :param xy_center:
    :param drizzled:
    :return:
    """
    x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20 = p

    M = np.matrix([[M_11, M_12],
                   [M_21, M_22]])
    if drizzled:
        M /= 2.0

    # pixel position must be WRT CCD center:
    if xy_center != (0, 0):
        x = x_raw - xy_center[0]
        y = y_raw - xy_center[1]
    else:
        x = x_raw
        y = y_raw

    delta_x = x - x_tan
    delta_y = y - y_tan
    # position on tangential plane in degrees
    uv = np.array(
        M * np.array([[delta_x + a_02 * delta_y ** 2 + a_11 * delta_x * delta_y + a_20 * delta_x ** 2],
                      [delta_y + b_02 * delta_y ** 2 + b_11 * delta_x * delta_y + b_20 * delta_x ** 2]]))

    return uv


def xy2radec(ra_tan, dec_tan, p, x_raw, y_raw, xy_center=(0, 0), drizzled=False):
    """

    :param ra_tan: tangent point RA/Dec
    :param dec_tan:
    :param p: astrometric solution
    :param x_raw:
    :param y_raw:
    :param xy_center:
    :param drizzled:
    :return:
    """

    # 3d position of the tangent point (on a unit sphere)
    t = np.array((np.cos(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0),
                  np.sin(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0),
                  np.sin(dec_tan * np.pi / 180.0)))

    # 3d basis vectors
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)

    uv = xy2uv(p, x_raw, y_raw, xy_center=xy_center, drizzled=drizzled)

    # R_i = t + np.pi / 180.0 * np.array([uv[0][0], uv[1][0], 0])
    R_i = t + np.pi / 180.0 * (uv[0][0] * u + uv[1][0] * v)
    r_i = R_i / np.linalg.norm(R_i)

    RA_i = 180.0 / np.pi * np.arctan2(np.dot(r_i, j), np.dot(r_i, i))
    if RA_i < 0:
        RA_i += 360.0
    Dec_i = 180.0 / np.pi * np.arcsin(np.dot(r_i, k))

    return RA_i, Dec_i


def nail_wcs(p, x_raw, y_raw, ra, dec, xy_center=(0, 0), drizzled=False):
    """
        Given Robo-AO KP astrometric parameters (linear mapping + quadratic distortion) and
        pixel position of known star (i.e. its RA/Dec are know from external sources),
        iteratively compute RA/Dec of the tangential point, thus 'nailing' the astrometric solution
        to the celestial sphere.
        Given RA_tan/Dec_tan, any pair (x_i, y_i) on the CCD can be converted to RA_i/Dec_i
    :param p:
    :param x_raw:
    :param y_raw:
    :param ra:
    :param dec:
    :param xy_center: x_raw and y_raw are centered around this point, e.g. (0, 0) or (512, 512)
    :param drizzled:
    :return:
    """

    star = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    uv = xy2uv(p, x_raw, y_raw, xy_center=xy_center, drizzled=drizzled)

    ra_0 = star.ra.deg + uv[0]
    dec_0 = star.dec.deg + uv[1]

    # iteratively find RA_tan, Dec_tan using multivariative Newton-Raphson's method
    r_star = np.array([np.cos(star.ra.rad) * np.cos(star.dec.rad),
                       np.sin(star.ra.rad) * np.cos(star.dec.rad),
                       np.sin(star.dec.rad)])

    def g(r, ra_tan, dec_tan):
        return r[0] * np.cos(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0) + \
               r[1] * np.sin(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0) + \
               r[2] * np.sin(dec_tan * np.pi / 180.0)

    def f1(r, ra_tan):
        return r[0] * np.sin(ra_tan * np.pi / 180.0) - \
               r[1] * np.cos(ra_tan * np.pi / 180.0)

    def f2(r, ra_tan, dec_tan):
        return -r[0] * np.cos(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0) - \
               r[1] * np.sin(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0) + \
               r[2] * np.cos(dec_tan * np.pi / 180.0)

    def dg_dra(r, ra_tan, dec_tan):
        return -r[0] * np.sin(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0) + \
               r[1] * np.cos(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0)

    def dg_ddec(r, ra_tan, dec_tan):
        return -r[0] * np.cos(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0) - \
               r[1] * np.sin(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0) + \
               r[2] * np.cos(dec_tan * np.pi / 180.0)

    def df1_dra(r, ra_tan):
        return r[0] * np.cos(ra_tan * np.pi / 180.0) + \
               r[1] * np.sin(ra_tan * np.pi / 180.0)

    def df2_dra(r, ra_tan, dec_tan):
        return r[0] * np.sin(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0) - \
               r[1] * np.cos(ra_tan * np.pi / 180.0) * np.sin(dec_tan * np.pi / 180.0)

    def df2_ddec(r, ra_tan, dec_tan):
        return -r[0] * np.cos(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0) - \
               r[1] * np.sin(ra_tan * np.pi / 180.0) * np.cos(dec_tan * np.pi / 180.0) - \
               r[2] * np.sin(dec_tan * np.pi / 180.0)

    x_tmp = np.zeros(2)
    x = np.array([ra_0, dec_0])

    # 1e-9 deg = 0.0036 mas
    while np.linalg.norm(x - x_tmp) > 1e-12:
        x_tmp = deepcopy(x)  # save previous iteration
        F = 180.0 / np.pi * np.array([f1(r_star, x[0])/g(r_star, x[0], x[1]),
                                      f2(r_star, x[0], x[1]) / g(r_star, x[0], x[1])]) - uv
        # print(F+uv, uv)

        J_11 = (df1_dra(r_star, x[0])*g(r_star, x[0], x[1]) -
                f1(r_star, x[0])*dg_dra(r_star, x[0], x[1])) / (g(r_star, x[0], x[1])**2)
        # print(J_11)

        J_12 = -f1(r_star, x[0]) * dg_ddec(r_star, x[0], x[1]) / (g(r_star, x[0], x[1]) ** 2)
        # print(J_12)

        J_21 = (df2_dra(r_star, x[0], x[1]) * g(r_star, x[0], x[1]) -
                f2(r_star, x[0], x[1]) * dg_dra(r_star, x[0], x[1])) / (g(r_star, x[0], x[1]) ** 2)
        # print(J_21)

        J_22 = (df2_ddec(r_star, x[0], x[1]) * g(r_star, x[0], x[1]) -
                f2(r_star, x[0], x[1]) * dg_ddec(r_star, x[0], x[1])) / (g(r_star, x[0], x[1]) ** 2)
        # print(J_22)

        J = 180.0 / np.pi * np.matrix([[J_11[0], J_12[0]],
                                       [J_21[0], J_22[0]]])

        x -= np.linalg.pinv(J) * F

        # print(x)

    return x


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Nail WCS given (x,y) and (RA,Dec) of single source')

    parser.add_argument('x', metavar='x', action='store', help='X pixel position [1, 1024]', type=float)
    parser.add_argument('y', metavar='y', action='store', help='Y pixel position [1, 1024]', type=float)
    parser.add_argument('ra', metavar='ra', action='store', help='RA [decimal degrees]', type=float)
    parser.add_argument('dec', metavar='dec', action='store', help='Dec [decimal degrees]', type=float)

    parser.add_argument('-d', action='store_true', dest='drizzled', help='input image x2 oversampled by Drizzle?')

    args = parser.parse_args()

    '''
    test case:
    x, y: [-391.9348999999999705  374.8141000000000531] 
    ra, dec: [ 250.4108537221999882   36.4594536398999978]
    
    ra_t, dec_t = 250.41571060683967   36.455930871780858
    '''

    # mapping parameters:
    # 20170620:
    # x_tan, y_tan, M_11, M_12, M_21, M_22, a_02, a_11, a_20, b_02, b_11, b_20
    mapping = np.array([-1.7097340244173049e+00, -2.9573349942689418e+00,
                        -9.9243124794318175e-06, 9.1942882975760912e-08,
                        -1.9338413987785716e-08, 9.7191812668049799e-06,
                        -7.3628182254640821e-07, -2.9041549472251136e-07, 2.6488080053108772e-07,
                        -5.5062280886481794e-05, 2.7321332070856352e-07, -5.3791223995525186e-05])

    # iteratively find RA/Dec of the tangential point
    ra_t, dec_t = nail_wcs(mapping, args.x, args.y, args.ra, args.dec, drizzled=args.drizzled)
    ra_t, dec_t = ra_t[0], dec_t[0]

    print(ra_t, dec_t)

    ''' Now, any (x, y) pair can be converted to (RA, Dec). For exmaple, '''
    ra_i, dec_i = xy2radec(ra_t, dec_t, mapping, 926.5996, 919.5908, xy_center=(512, 512), drizzled=False)
    ra_i, dec_i = xy2radec(ra_t, dec_t, mapping, -391.9349, 374.8141, xy_center=(0, 0), drizzled=False)

    print(ra_i, dec_i)
