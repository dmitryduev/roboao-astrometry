# coding=utf-8
"""
Red lights, green lights, strawberry wine
A good friend of mine
Follows the stars,
Venus and Mars are alright tonight!

Produce a _nightly_ list of observable planets/moons brighter than 16 mag
for the Robo-AO queue

@autor: Dr Dmitry A. Duev [Caltech]


"""

import datetime
from solarsyslib2 import TargetListComets#, TargetXML
import pytz
import os
import inspect
import ConfigParser
import argparse

if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(prog='comets_tonight.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Create nightly target list for the comets program.')
    # positional arguments
    parser.add_argument('config_file', type=str, help='config file')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()

    # load config data
    config = ConfigParser.RawConfigParser()
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    try:
        if not os.path.isabs(args.config_file):
            config.read(os.path.join(abs_path, args.config_file))
        else:
            config.read(args.config_file)
    except IOError:
        print('failed to load config file {:s}, trying ./config.ini'.format(args.config_file))
        try:
            config.read(os.path.join(abs_path, 'config.ini'))
        except IOError:
            raise Exception('config file ./config.ini not found')

    f_inp = config.get('Path', 'pypride_inp')

    # observatory and time zone
    observatory = config.get('Observatory', 'observatory')
    timezone = config.get('Observatory', 'timezone')

    # observability settings:
    # nighttime between twilights: astronomical (< -18 deg), civil (< -6 deg), nautical (< -12 deg)
    twilight = config.get('Comets', 'twilight')
    # fraction of night when observable given constraints:
    fraction = float(config.get('Comets', 'fraction'))
    # magnitude limit:
    m_lim = float(config.get('Comets', 'm_lim'))
    # elevation cut-off [deg]:
    elv_lim = float(config.get('Comets', 'elv_lim'))

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone(timezone))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)
    print('\nstarted at:', datetime.datetime.now(pytz.timezone(timezone)))

    tl = TargetListComets(f_inp, database_source='mpc', database_file='CometEls.txt',
                          _observatory=observatory, _m_lim=m_lim, _elv_lim=elv_lim, _date=today)
    # tl.target_list_all(today)
    # targets = tl.target_list_observable(tl.target_list_all(today), today,
    #                                     elv_lim=elv_lim, twilight=twilight, fraction=fraction)
    #
    # ''' make/change XML files '''
    # path = config.get('Path', 'program_path')
    # program_number = config.get('Path', 'program_number_comets')
    # txml = TargetXML(path=path, program_number=program_number,
    #                  server=config.get('Path', 'queue_server'))
    # # dump 'em targets!
    # c = txml.dumpTargets(targets, epoch='J2000')
    # if c is None:
    #     print('Successfully updated the target list via the website')
    #
    #     # clean up the target list - remove unobserved, which are not suitable anymore:
    #     txml.clean_target_list()
