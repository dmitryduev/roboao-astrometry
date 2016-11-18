from __future__ import print_function
import numpy as np
import os
import datetime
import urllib2


class AsteroidDatabase(object):

    def __init__(self):
        self.database = None
        self.database_url = None
        self.f_database = None

    @staticmethod
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

    def load(self):
        """
            Load database into self.database
        :return:
        """
        raise NotImplementedError

    def get_one(self):
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

    def __init__(self):
        super(AsteroidDatabaseJPL, self).__init__()

    def load(self):
        raise NotImplementedError

    def get_one(self):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError


class AsteroidDatabaseMPC(AsteroidDatabase):

    def __init__(self):
        super(AsteroidDatabaseMPC, self).__init__()

    def load(self):
        raise NotImplementedError

    def get_one(self):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError


class Asteroid(object):

    def __init__(self):
        pass




if __name__ == '__main__':
    pass
