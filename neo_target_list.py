from __future__ import print_function
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import Element
from dicttoxml import dicttoxml
import json
from collections import OrderedDict
import os
import datetime


def make_target(program_number=444, target_number=1, name=None,
                comment=None, t_start=None, t_stop=None,
                ra='00:00:00', dec='+00:00:00', epoch=2000.0, mag=12, exp=120,
                filter_code='FILTER_LONGPASS_600', camera_mode=10):
    target = OrderedDict([('program_number', program_number),
                          ('number', target_number),
                          ('name', name),
                          ('visited_times_for_completion', '1'),
                          ('seeing_limit', ''),
                          ('visited_times', '0'),
                          ('done', '0'),
                          ('cadence', '0'),
                          ('comment', comment),
                          ('time_critical_start', t_start),
                          ('time_critical_stop', t_stop),
                          ('Object', [])])
    target['Object'].append(OrderedDict([('number', '1'),
                                         ('RA', ra), ('dec', dec),
                                         ('epoch', epoch), ('magnitude', mag),
                                         ('sun_altitude_limit', ''), ('moon_phase_window', ''),
                                         ('airmass_limit', ''), ('sun_distance_limit', ''),
                                         ('moon_distance_limit', ''), ('sky_brightness_limit', ''),
                                         ('hour_angle_limit', ''), ('done', '0'),
                                         ('Observation', [])]))
    target['Object'][0]['Observation'].append(
                    OrderedDict([('number', '{:d}'.format(1)),
                                 ('exposure_time', exp),
                                 ('ao_flag', '1'),
                                 ('filter_code', filter_code),
                                 ('camera_mode', camera_mode),
                                 ('repeat_times', '1'),
                                 ('repeated', '0'),
                                 ('done', '0')]))

    return target


def make_xml(target, path):
    # build an xml-file:

    target_xml = dicttoxml(target, custom_root='Target', attr_type=False)
    # this is good enough, but adds unnecessary <item> tags. remove em:
    dom = minidom.parseString(target_xml)
    target_xml = dom.toprettyxml()
    # <item>'s left extra \t's after them - remove them:
    target_xml = target_xml.replace('\t\t\t', '\t\t')
    target_xml = target_xml.replace('\t\t\t\t', '\t\t\t')
    target_xml = target_xml.replace('<?xml version="1.0" ?>', '')
    target_xml = target_xml.split('\n')
    target_xml = [t for t in target_xml if 'item>' not in t]
    # deal with missing <Object>s and <Observation>s:
    #        xml_out = []
    #        for line in target_xml[1:-1]:
    #            xml_out.append('{:s}\n'.format(line))
    #        xml_out.append('{:s}'.format(target_xml[-1]))
    ind_obs_start = [i for i, v in enumerate(target_xml) if '<Observation>' in v]
    ind_obs_stop = [i for i, v in enumerate(target_xml) if '</Observation>' in v]
    for (start, stop) in zip(ind_obs_start, ind_obs_stop):
        ind_num_obs = [i + start for i, v in enumerate(target_xml[start:stop]) if '<number>' in v]
        if len(ind_num_obs) > 1:
            for ind in ind_num_obs[:0:-1]:
                target_xml.insert(ind, '\t\t</Observation>\n\t\t<Observation>')

    ind_obj = [i for i, v in enumerate(target_xml) if v[:10] == '\t\t<number>']
    for ind in ind_obj[:0:-1]:
        target_xml.insert(ind, '\t</Object>\n\t<Object>')

    #        print target_xml

    target_xml_path = os.path.join(path, 'Target_{:d}.xml'.format(target['number']))

    target_xml = [_t for _t in target_xml if '/>' not in _t]
    with open(target_xml_path, 'w') as _f:
        for line in target_xml[1:-1]:
            _f.write('{:s}\n'.format(line))
        _f.write('{:s}'.format(target_xml[-1]))


if __name__ == '__main__':

    date0 = datetime.datetime.utcnow()
    date = datetime.datetime(date0.year, date0.month, date0.day) + datetime.timedelta(days=1)
    # date = datetime.datetime(date0.year, date0.month, date0.day)

    # date = datetime.datetime(2018, 4, 27)

    path_nightly = '/home/roboao/dev/roboao-astrometry/obs/'
    # path_nightly = '/Users/dmitryduev/_caltech/python/roboao-astrometry/obs/'

    asteroid_class = 'pha'

    path_nightly_date = os.path.join(path_nightly, date.strftime('%Y%m%d'), asteroid_class)

    tonight_json = os.path.join(path_nightly_date, 'bright_objects.json')
    if os.path.exists(tonight_json):
        with open(tonight_json, 'r') as f:
            neos = json.load(f)

        tn = 1

        for neo in neos:
            if len(neos[neo]['guide_stars']) > 0:
                print(neo)

                for gs in neos[neo]['guide_stars']:
                    nm = gs['id']
                    cm = '{:s}__{:s}m@{:s}arcsecMinDist'.format(neo, neos[neo]['mean_magnitude'],
                                                                  gs['min_separation'])
                    # (194126)_2001_SG276__15.764m@12.814arcsecMinDist
                    mg = gs['magnitude']

                    # FIXME:
                    if float(mg) > 14.0:
                        continue

                    tstart = datetime.datetime.strptime(gs['obs_window'][0], '%Y%m%d_%H%M%S.%f')
                    tstop = datetime.datetime.strptime(gs['obs_window'][1], '%Y%m%d_%H%M%S.%f')

                    t = make_target(program_number=444, target_number=tn, name=nm, comment=cm,
                                    t_start=(tstart -
                                             datetime.timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S.%f'[:-3]),
                                    t_stop=tstop.strftime('%Y-%m-%d %H:%M:%S.%f'[:-3]),
                                    ra=gs['radec'][0], dec=gs['radec'][1],
                                    epoch=2000.0, mag=mg, exp=120,
                                    filter_code='FILTER_LONGPASS_600', camera_mode=10)
                    # print(t)

                    make_xml(target=t, path=path_nightly_date)

                    tn += 1
