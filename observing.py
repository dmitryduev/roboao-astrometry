from __future__ import print_function
from gevent import monkey
monkey.patch_all()

import os
import json
import datetime
import ConfigParser
import flask
import inspect
import traceback
from collections import OrderedDict


def daterange(start_date, end_date):
    """
        Helper to iterate over a range of dates
    :param start_date:
    :param end_date:
    :return:
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def get_config(config_file='config.ini'):
    """
        load config data
    """
    try:
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        _config = ConfigParser.RawConfigParser()
        _config.read(os.path.join(abs_path, config_file))
        # logger.debug('Successfully read in the config file {:s}'.format(args.config_file))

        ''' connect to mongodb database '''
        conf = dict()
        # paths:
        conf['path_nightly'] = _config.get('Path', 'path_nightly')

        ''' server location '''
        conf['server_host'] = _config.get('Server', 'host')
        conf['server_port'] = _config.get('Server', 'port')

        return conf
    except Exception as e:
        print(e)
        traceback.print_exc()

''' initialize the Flask app '''
app = flask.Flask(__name__)
app.secret_key = 'roboaokicksass'

''' get config data '''
config = get_config(config_file='config.ini')


@app.route('/data/<path:filename>')
def data_static(filename):
    """
        Get files from the archive
    :param filename:
    :return:
    """
    _p, _f = os.path.split(filename)
    return flask.send_from_directory(os.path.join(config['path_nightly'], _p), _f)


def stream_template(template_name, **context):
    """
        see: http://flask.pocoo.org/docs/0.11/patterns/streaming/
    :param template_name:
    :param context:
    :return:
    """
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


# serve root
@app.route('/', methods=['GET'])
def root():

    if 'start' in flask.request.args:
        start = flask.request.args['start']
    else:
        start = None
    if 'stop' in flask.request.args:
        stop = flask.request.args['stop']
    else:
        stop = None

    def iter_dates(_dates):
        """
            instead of first loading and then sending everything to user all at once,
             yield data for a single date at a time and stream to user
        :param _dates:
        :return:
        """
        if len(_dates) > 0:
            for _date in _dates:
                # print(_date, _dates[_date])
                yield _date, _dates[_date]
        else:
            yield None, None

    # get all dates:
    dates = get_dates(start=start, stop=stop)

    return flask.Response(stream_template('template-observing.html',
                                          dates=iter_dates(dates)))


def get_dates(start=None, stop=None):
    if start is None:
        # this is ~when we moved to KP:
        start = datetime.datetime(2016, 12, 1)
        # # by default -- last 30 days:
        # start = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    else:
        try:
            start = datetime.datetime.strptime(start, '%Y%m%d')
        except Exception as _e:
            print(_e)
            # start = datetime.datetime.utcnow() - datetime.timedelta(days=10)
            start = datetime.datetime(2016, 12, 1)

    if stop is None:
        stop = datetime.datetime.utcnow() + datetime.timedelta(days=30)
    else:
        try:
            stop = datetime.datetime.strptime(stop, '%Y%m%d')
        except Exception as _e:
            print(_e)
            stop = datetime.datetime.utcnow() + datetime.timedelta(days=30)

    dates = OrderedDict()

    for date in daterange(start, stop):
        date_str = date.strftime('%Y%m%d')
        # load JSON file with the date's data
        f_json_sci = os.path.join(config['path_nightly'], '{:s}'.format(date_str), 'bright_objects.json')

        if os.path.exists(f_json_sci):
            try:
                with open(f_json_sci) as fjson_sci:
                    data = json.load(fjson_sci, object_pairs_hook=OrderedDict)
                    data = OrderedDict(data)
                    dates[date_str] = data
            except Exception as e:
                print(e)

    return dates


@app.errorhandler(500)
def internal_error(error):
    return '500 error'


@app.errorhandler(404)
def not_found(error):
    return '404 error'


@app.errorhandler(403)
def not_found(error):
    return '403 error: forbidden'


if __name__ == '__main__':
    app.run(host=config['server_host'], port=config['server_port'], threaded=True)
