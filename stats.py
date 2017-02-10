from __future__ import print_function
import os
import json

if __name__ == '__main__':
    path_obs = 'obs'
    f_json = 'bright_objects.json'
    asteroid_class = 'pha'

    dates = [d for d in os.listdir(path_obs) if os.path.isdir(os.path.join(path_obs, d))]
    # print(dates)

    objects = set()

    for d in dates:
        try:
            with open(os.path.join(path_obs, d, asteroid_class, f_json)) as json_data:
                obj_date = json.load(json_data)
                # print(obj_date.keys())
                for obj in obj_date.keys():
                    if obj_date[obj]['is_observable'] == 'true' and len(obj_date[obj]['guide_stars']) > 0:
                        objects.add(obj)
        except Exception as e:
            print(d, e)
            continue

    print(objects)
    print(len(objects))
