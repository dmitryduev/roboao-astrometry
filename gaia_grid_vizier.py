from __future__ import print_function
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import astropy.units as u
import astropy.coordinates as coord
from astropy import wcs
from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter

# import matplotlib
# matplotlib.use('Qt5Agg')

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


def generate_image(xy, mag, nx=2048, ny=2048):
    """

    :param xy:
    :param mag:
    :param nx:
    :param ny:
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

    # Convolve with a gaussian
    image = gaussian_filter(image, 7)

    return image


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

    # create a chart
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

    sim_image = generate_image(xy=pix_stars, mag=mag_stars, nx=w.naxis1, ny=w.naxis2)

    # convert to fits hdu:
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

    # display field
    # fig.show_colorscale(cmap='viridis')  # magma
    fig.show_grayscale()
    # fig.show_markers(grid_stars[cat]['_RAJ2000'], grid_stars[cat]['_DEJ2000'],
    #                  layer='marker_set_1', edgecolor='white',
    #                  facecolor='white', marker='o', s=30, alpha=0.7)

    plt.show()
