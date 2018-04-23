import string
import random

import requests
from socketIO_client import SocketIO
from bson.json_util import loads
from concurrent.futures import ThreadPoolExecutor

import numpy as np


''' PENQUINS - Processing ENormous Queries of ztf Users INStantaneously '''


def radec_str2rad(_ra_str, _dec_str):
    """

    :param _ra_str: 'H:M:S'
    :param _dec_str: 'D:M:S'
    :return: ra, dec in rad
    """
    # convert to rad:
    _ra = list(map(float, _ra_str.split(':')))
    _ra = (_ra[0] + _ra[1] / 60.0 + _ra[2] / 3600.0) * np.pi / 12.
    _dec = list(map(float, _dec_str.split(':')))
    _sign = np.sign(_dec[0]) if _dec[0] != 0 else 1
    _dec = _sign * (abs(_dec[0]) + abs(_dec[1]) / 60.0 + abs(_dec[2]) / 3600.0) * np.pi / 180.

    return _ra, _dec


def radec_str2geojson(ra_str, dec_str):

    # hms -> ::, dms -> ::
    if isinstance(ra_str, str) and isinstance(dec_str, str):
        if ('h' in ra_str) and ('m' in ra_str) and ('s' in ra_str):
            ra_str = ra_str[:-1]  # strip 's' at the end
            for char in ('h', 'm'):
                ra_str = ra_str.replace(char, ':')
        if ('d' in dec_str) and ('m' in dec_str) and ('s' in dec_str):
            dec_str = dec_str[:-1]  # strip 's' at the end
            for char in ('d', 'm'):
                dec_str = dec_str.replace(char, ':')

        if (':' in ra_str) and (':' in dec_str):
            ra, dec = radec_str2rad(ra_str, dec_str)
            # convert to geojson-friendly degrees:
            ra = ra * 180.0 / np.pi - 180.0
            dec = dec * 180.0 / np.pi
        else:
            raise Exception('Unrecognized string ra/dec format.')
    else:
        # already in degrees?
        ra = float(ra_str)
        # geojson-friendly ra:
        ra -= 180.0
        dec = float(dec_str)

    return ra, dec


class Query(object):
    """
        Query object
    """

    def __init__(self, query=None, protocol='https', host='kowalski.caltech.edu', port=443,
                 user=None, access_token=None, verbose=False):

        assert query is not None, 'query must be specified'
        assert user is not None, 'user must be specified'
        assert access_token is not None, 'access_token must be specified'

        # Kowalski, status!
        self.v = verbose

        self.protocol = protocol

        self.host = host
        self.port = port

        self.user = user
        self.access_token = access_token

        self.query = query

        # generate a unique hash id and store it in query (for further query identification)
        _id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32)).lower()
        if 'kwargs' not in self.query:
            self.query['kwargs'] = dict()
        self.query['kwargs']['_id'] = _id

        # place holders
        self._socket = None
        self._task_id = None
        self._result = None

    def execute(self, timeout=60):
        # for now had to limit transports to xhr-polling as websocket sometimes has an issue with TLS
        if self._socket is None:
            self._socket = SocketIO('{:s}://{:s}'.format(self.protocol, self.host),
                                    self.port, wait_for_connection=False,
                                    headers={'Authorization': 'Bearer {:s}'.format(self.access_token)},
                                    transports=['xhr-polling'])

        # print(q)
        self._socket.emit('query', self.query)

        # register event listeners
        self._socket.on('msg', self.on_msg)
        self._socket.on('enqueued', self.on_enqueued)
        self._socket.on('finished', self.on_finished)
        self._socket.on('post_query_result_view', self.on_post_query_result_view)

        if timeout == 0:
            self._socket.wait()
        else:
            self._socket.wait(seconds=timeout)

    def on_enqueued(self, msg):
        try:
            _msg = loads(msg['task_doc_msg'])
            # query successfully enqueued? save _id:
            _id = _msg['kwargs']['_id'] if (('kwargs' in _msg) and ('_id' in _msg['kwargs'])) else None

            if _id == self.query['kwargs']['_id']:
                # save task id:
                task_id = msg['task_id']
                self._task_id = task_id
                if self.v:
                    print('task enqueued:', msg)
        except Exception as e:
            # close socket connection:
            if self._socket is not None:
                self._socket.disconnect()
            if self.v:
                print('Something went wrong: {:s}'.format(str(e)))

    def on_finished(self, msg):
        try:
            # get task id:
            task_id = msg['task_id']
            if task_id == self._task_id:
                if self.v:
                    print('task finished:', msg)
                # emit get_query_result_view event
                # (get_query_result will trigger fetching the file in the browser, if kowalski.caltech.edu is open :)
                self._socket.emit('get_query_result_view', task_id)

        except Exception as e:
            if self.v:
                print('Something went wrong: {:s}'.format(str(e)))

    def on_post_query_result_view(self, msg):
        try:
            # get task id:
            task_id = msg['task_id']
            if task_id == self._task_id:
                if self.v:
                    print(msg)
                # fetch and unpack result
                self._result = dict()
                self._result['result'] = loads(msg['msg'])
                self._result['task_id'] = msg['task_id']

                if self.v:
                    print(self._result)
                # shut down socket connection
                if self._socket is not None:
                    self._socket.disconnect()
                    self._socket = None

        except Exception as e:
            if self.v:
                print('Something went wrong: {:s}'.format(str(e)))

    def on_msg(self, msg):
        try:
            # get task id:
            task_id = msg['task_id']
            if task_id == self._task_id:
                print('received message:', msg)

                if 'already exists' in msg['msg']:
                    self._result = dict()
                    self._result['result'] = None
                    self._result['task_id'] = msg['task_id']
                    # shut down socket connection
                    if self._socket is not None:
                        self._socket.disconnect()
                        self._socket = None

        except Exception as e:
            if self.v:
                print('Something went wrong: {:s}'.format(str(e)))


class Kowalski(object):
    """
        Query ZTF TDA databases
    """

    def __init__(self, protocol='https', host='kowalski.caltech.edu', port=443, verbose=False,
                 username=None, password=None):

        assert username is not None, 'username must be specified'
        assert password is not None, 'password must be specified'

        # status, Kowalski!
        self.v = verbose

        self.protocol = protocol

        self.host = host
        self.port = port

        self.username = username
        self.password = password

        self.access_token = self.authenticate()

        # init place holders for threaded operations
        self._pool = None
        self._futures = []

    def authenticate(self):
        access_token = None

        try:
            # post username and password, get access token
            auth = requests.post('{:s}://{:s}:{:d}/auth'.format(self.protocol, self.host, self.port),
                                 json={"username": self.username, "password": self.password})
            if self.v:
                print(auth.json())

            access_token = auth.json()['access_token']

            if self.v:
                print('Successfully authenticated')

            return access_token

        except Exception as e:
            if self.v:
                print(str(e))
            #     print('Authentication failed: {:s}'.format(str(e)))
            print('Authentication failed')

        finally:
            return access_token

    def check_connection(self):
        try:
            _r = requests.get('{:s}://{:s}:{:d}/protected'.format(self.protocol, self.host, self.port),
                              headers={'Authorization': 'Bearer {:s}'.format(self.access_token)})
            if self.v:
                print(_r)

            _r = requests.get('{:s}://{:s}:{:d}/partially-protected'.format(self.protocol, self.host, self.port),
                              headers={'Authorization': 'Bearer {:s}'.format(self.access_token)})
            if self.v:
                print(_r)

            print('Connection is OK.')
        except Exception as e:
            if self.v:
                print('Something is wrong with the connection: {:s}'.format(str(e)))

    def query_sync(self, query, timeout=60):
        """

        :param query:
        :param timeout: give up and disconnect after that many seconds; wait forever if 0
        :return:
        """

        assert self.access_token is not None, 'authenticate first before querying'

        try:
            _q = Query(query=query, protocol=self.protocol, host=self.host, port=self.port,
                       user=self.username, access_token=self.access_token, verbose=self.v)
            _q.execute(timeout=timeout)

            return _q._result

        except OSError:
            print('Error while running query. Try again later.')

            try:
                # maybe your token expired?
                self.access_token = self.authenticate()

                # now try again
                _q = Query(query=query, protocol=self.protocol, host=self.host, port=self.port,
                           user=self.username, access_token=self.access_token, verbose=self.v)
                _q.execute(timeout=timeout)

                return _q._result

            finally:
                pass

            return None

    def query_async(self, query, timeout=60):

        assert self.access_token is not None, 'authenticate first before querying'

        future = None

        try:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=4)

            future = self._pool.submit(self.query_sync, query, timeout)
            # self._futures.append(future)
        except Exception as e:
            if self.v:
                print('Something went horribly wrong: {:s}'.format(str(e)))
        finally:
            return future
