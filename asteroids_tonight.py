from __future__ import print_function
import argparse
import os
import ConfigParser
import inspect
import datetime
import pytz
from solarsyslib2 import TargetListAsteroids

if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='UTC date %Y%m%d. default: upcoming night in Arizona', type=str)

    parser.add_argument('--asteroid_class', metavar='asteroid_class', action='store', dest='asteroid_class',
                        help='asteroid class: nea, pha, daily, temporary', type=str, default='nea')

    args = parser.parse_args()
    config_file = args.config_file
    asteroid_class = args.asteroid_class

    if asteroid_class == 'pha':
        database_file = 'PHA.txt'
    elif asteroid_class == 'nea':
        database_file = 'NEA.txt'
    elif asteroid_class == 'daily':
        database_file = 'DAILY.DAT'
    else:
        print('could not recognize asteroid class, falling back to NEA')
        asteroid_class = 'nea'
        database_file = 'NEA.txt'

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

    ''' Target list '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    if not args.date:
        now = datetime.datetime.now(pytz.timezone(timezone))
        # now = datetime.datetime.utcnow()
        date = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)
    else:
        try:
            date = datetime.datetime.strptime(args.date, '%Y%m%d')
        except Exception as e:
            print(e)
            print('failed to parse date, running for tonight at Kitt Peak')
            now = datetime.datetime.now(pytz.timezone(timezone))
            # now = datetime.datetime.utcnow()
            date = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

    print('\nrunning computation for:', date)
    print('\nstarted at:', datetime.datetime.now(pytz.timezone(timezone)))

    tl = TargetListAsteroids(f_inp, database_source='mpc', database_file=database_file,
                             _observatory=observatory, _m_lim=m_lim, _elv_lim=elv_lim, _date=date)
    # get all bright targets given m_lim and check observability given elv_lim, twilight and fraction
    mask = None
    tl.target_list(date, mask, _parallel=True, _epoch='J2000', _output_Vmag=True, _night_grid_n=40,
                   _twilight=twilight, _fraction=fraction)

    for a in tl.targets:
        print(a)
