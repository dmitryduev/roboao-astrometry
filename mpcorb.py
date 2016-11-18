from __future__ import print_function
import numpy as np
import os
import urllib2
import datetime
from pypride.vintlib import mjuliandate


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


def asteroid_database_update(_f_database, _url, n=1.0):
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
            print('Asteroid database: {:s} is out of date, updating...'.format(_f_database))
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


def asteroid_database_load(_f_database, _provider='mpc'):
    """
        Load MPC database
        :param _provider: 'mpc' or 'jpl'
    """

    with open(_f_database, 'r') as f:
        database = f.readlines()

    start = [i for i, l in enumerate(database[:300]) if l[0:2] == '--']
    if len(start) > 0:
        database = database[start[0] + 1:]

    if _provider == 'jpl':
        dt = np.dtype([('num', '<i8'), ('name', '|S21'),
                       ('epoch', '<i8'), ('a', '<f8'),
                       ('e', '<f8'), ('i', '<f8'),
                       ('w', '<f8'), ('Node', '<f8'),
                       ('M0', '<f8'), ('H', '<f8'), ('G', '<f8')])
        return np.array([((int(l[0:6]),) + (l[6:25].strip(),) +
                          tuple(map(float, l[25:].split()[:-2]))) for l in database[2:]],
                        dtype=dt)

    elif _provider == 'mpc':
        dt = np.dtype([('designation', '|S21'), ('H', '<f8'), ('G', '<f8'),
                       ('epoch', '<f8'), ('M0', '<f8'), ('w', '<f8'),
                       ('Node', '<f8'), ('i', '<f8'), ('e', '<f8'),
                       ('n', '<f8'), ('a', '<f8'), ('U', '|S21'),
                       ('n_obs', '<f8'), ('n_opps', '<f8'), ('arc', '|S21'),
                       ('rms', '<f8'), ('name', '|S21'), ('last_obs', '|S21')
                      ])

        return np.array([(str(entry[:7]).strip(),) + (float(entry[8:13]),) + (float(entry[14:19]),)
                           + (unpack_epoch(str(entry[20:25])),) + (float(entry[26:35]),) + (float(entry[37:46]),)
                           + (float(entry[48:57]),) + (float(entry[59:68]),) + (float(entry[70:79]),)
                           + (float(entry[80:91]),) + (float(entry[92:103]),) + (str(entry[105:106]),)
                           + (int(entry[117:122]),) + (int(entry[123:126]),) + (str(entry[127:136]).strip(),)
                           + (float(entry[137:141]),) + (str(entry[166:194]).strip().replace(' ', '_'),)
                           + (str(entry[194:202]).strip(),)
                         for entry in database], dtype=dt)


if __name__ == '__main__':
    f_database = 'PHA.txt'
    url = 'http://www.minorplanetcenter.net/iau/MPCORB/PHA.txt'
    asteroid_database_update(_f_database=f_database, _url=url, n=0.1)

    database = asteroid_database_load(_f_database=f_database, _provider='mpc')
    print(database)
    print(len(database))
