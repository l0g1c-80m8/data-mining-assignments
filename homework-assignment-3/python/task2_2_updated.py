import json
import pandas as pd
import os
import sys
import xgboost as xgb
import numpy as np

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'competition_project'
    run_time_params['in_dir'] = sys.argv[1]
    run_time_params['test_file'] = sys.argv[2]
    run_time_params['out_file'] = sys.argv[3]
    run_time_params['train'] = 'yelp_train.csv'
    run_time_params['val'] = 'yelp_val.csv'
    run_time_params['user'] = 'user.json'
    run_time_params['checkin'] = 'checkin.json'
    run_time_params['photo'] = 'photo.json'
    run_time_params['tip'] = 'tip.json'
    run_time_params['business'] = 'business.json'
    run_time_params['record_cols'] = ['user_id', 'business_id', 'rating']
    run_time_params['user_feature_cols'] = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']
    run_time_params['business_feature_cols'] = ['business_stars', 'business_review_count', 'business_is_open',
                                                'business_city', 'business_latitude', 'business_longitude',
                                                'business_state'
                                                ]
    return run_time_params


def parse_train_set():
    filename = '{}/{}'.format(params['in_dir'], params['train'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_val_set():
    filename = '{}/{}'.format(params['in_dir'], params['val'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_test_set():
    with open(params['test_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['test_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1]))


def parse_checkin_count():
    return sc.textFile('{}/{}'.format(params['in_dir'], params['checkin'])) \
        .map(lambda line: json.loads(line)) \
        .map(lambda checkin_obj: (checkin_obj['business_id'], sum(checkin_obj['time'].values())))


def parse_business_photo_count():
    return sc.textFile('{}/{}'.format(params['in_dir'], params['photo'])) \
        .map(lambda line: json.loads(line)) \
        .map(lambda photo_obj: (photo_obj['business_id'], photo_obj['photo_id'])) \
        .groupByKey() \
        .map(lambda business_id_count: (business_id_count[0], len(business_id_count[1])))


def parse_business_tip_count():
    return sc.textFile('{}/{}'.format(params['in_dir'], params['tip'])) \
        .map(lambda line: json.loads(line)) \
        .map(lambda tip_obj: (tip_obj['business_id'], tip_obj['user_id'])) \
        .groupByKey() \
        .map(lambda business_id_count: (business_id_count[0], len(business_id_count[1])))


def parse_user_tip_count():
    return sc.textFile('{}/{}'.format(params['in_dir'], params['tip'])) \
        .map(lambda line: json.loads(line)) \
        .map(lambda tip_obj: (tip_obj['user_id'], tip_obj['business_id'])) \
        .groupByKey() \
        .map(lambda business_id_count: (business_id_count[0], len(business_id_count[1])))


def parse_user_set(user_tip_count):
    filename = '{}/{}'.format(params['in_dir'], params['user'])

    def _user_features(user_obj):
        features = tuple(map(
            lambda col_name: user_obj[col_name],
            params['user_feature_cols']
        ))
        if user_obj['friends'] is None:
            features += (None,)
        else:
            features += (len(user_obj['friends'].split(', ')),)
        features += (sum(map(
            lambda val: int(val),
            [
                user_obj.get('compliment_hot', 0),
                user_obj.get('compliment_more', 0),
                user_obj.get('compliment_profile', 0),
                user_obj.get('compliment_cute', 0),
                user_obj.get('compliment_list', 0),
                user_obj.get('compliment_note', 0),
                user_obj.get('compliment_plain', 0),
                user_obj.get('compliment_cool', 0),
                user_obj.get('compliment_funny', 0),
                user_obj.get('compliment_writer', 0),
                user_obj.get('compliment_photos', 0),
            ]
        )),)
        features += (user_tip_count.get(user_obj['user_id'], 0), )

        return user_obj[params['record_cols'][0]], features

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(_user_features)


def parse_business_set(business_checkin_count, business_photo_count, business_tip_count):
    filename = '{}/{}'.format(params['in_dir'], params['business'])

    def _business_features(business_obj):
        features = tuple(map(
            lambda col_name: business_obj[col_name.replace('business_', '')],
            params['business_feature_cols']
        ))
        if business_obj['categories'] is None:
            features += (0,)
        else:
            features += (len(business_obj['categories'].split(', ')),)
        if business_obj['attributes'] is None:
            features += tuple([None] * 9)
        else:
            features += tuple(map(
                lambda val: int(bool(val)) if val is not None else None,
                (
                    business_obj['attributes'].get('Alcohol', None),
                    business_obj['attributes'].get('RestaurantsDelivery', None),
                    business_obj['attributes'].get('GoodForKids', None),
                    business_obj['attributes'].get('OutdoorSeating', None),
                    business_obj['attributes'].get('RestaurantsGoodForGroups', None),
                    business_obj['attributes'].get('RestaurantsTableService', None),
                    business_obj['attributes'].get('RestaurantsTakeOut', None),
                    business_obj['attributes'].get('Caters', None),
                    business_obj['attributes'].get('WheelchairAccessible', None),
                )))
            features += (
                business_obj['attributes'].get('RestaurantsPriceRange2', None),
                business_checkin_count.get(business_obj['business_id'], 0),
                business_photo_count.get(business_obj['business_id'], 0),
                business_tip_count.get(business_obj['business_id'], 0)
            )

        return business_obj[params['record_cols'][1]], features

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(_business_features)


def write_results_to_file(data):
    file_header = '{}\n'.format(', '.join(params['record_cols']))
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for record in data:
            fh.write('{}\n'.format(','.join(map(lambda item: str(item), list(record)))))


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], tuple([0] * 9))
    business_features = business_data.get(record[1], tuple([0] * 21))
    return record + user_features + business_features


def main():
    # create feature data
    user_tip_count = parse_user_tip_count().collectAsMap()
    user_data = parse_user_set(user_tip_count).collectAsMap()
    business_checkin_count = parse_checkin_count().collectAsMap()
    business_photo_count = parse_business_photo_count().collectAsMap()
    business_tip_count = parse_business_tip_count().collectAsMap()
    business_data = parse_business_set(business_checkin_count, business_photo_count, business_tip_count).collectAsMap()

    # create training dataset
    train_df = pd.DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=params['record_cols'] + params['user_feature_cols'] + ['n_friends', 'n_compliments', 'tip_count']
                + params['business_feature_cols']
                + ['n_cats', 'alcohol', 'delivery', 'kids', 'seating', 'groups', 'table_service', 'takeout',
                   'caters', 'wheelchair', 'price_range', 'checkin_count', 'photo_count', 'tip_count']
    )
    # train_df = train_df.append(pd.DataFrame(
    #     parse_val_set()
    #     .map(lambda record: fill_features(record, user_data, business_data))
    #     .collect(),
    #     columns=params['record_cols'] + params['user_feature_cols'] + ['n_friends', 'n_compliments', 'tip_count']
    #             + params['business_feature_cols']
    #             + ['n_cats', 'alcohol', 'delivery', 'kids', 'seating', 'groups', 'table_service', 'takeout',
    #                'caters', 'wheelchair', 'price_range', 'checkin_count', 'photo_count', 'tip_count']
    # ))
    train_df = train_df.fillna(value=np.nan)

    # create test dataset
    test_set = parse_test_set() \
        .map(lambda record: fill_features(record, user_data, business_data)) \
        .collect()
    test_df = pd.DataFrame(test_set,
                           columns=params['record_cols'][: -1] + params['user_feature_cols']
                                   + ['n_friends', 'n_compliments', 'tip_count']
                                   + params['business_feature_cols']
                                   + ['n_cats', 'alcohol', 'delivery', 'kids', 'seating', 'groups', 'table_service',
                                      'takeout', 'caters', 'wheelchair', 'price_range', 'checkin_count', 'photo_count',
                                      'tip_count']
                           )
    test_df = test_df.fillna(value=np.nan)

    le = LabelEncoder()
    le.fit(train_df['business_city'].astype(str))
    test_df['business_city'] = test_df['business_city'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    train_df['business_city'] = le.transform(train_df['business_city'].astype(str))
    test_df['business_city'] = le.transform(test_df['business_city'].astype(str))
    le = LabelEncoder()
    le.fit(train_df['business_state'].astype(str))
    test_df['business_state'] = test_df['business_state'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    train_df['business_state'] = le.transform(train_df['business_state'].astype(str))
    test_df['business_state'] = le.transform(test_df['business_state'].astype(str))
    le = LabelEncoder()
    le.fit(train_df['price_range'].astype(str))
    test_df['price_range'] = test_df['price_range'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    train_df['price_range'] = le.transform(train_df['price_range'].astype(str))
    test_df['price_range'] = le.transform(test_df['price_range'].astype(str))

    train_x, train_y = train_df.drop(params['record_cols'], axis=1).values, train_df[params['record_cols'][-1:]].values
    test_x = test_df.drop(params['record_cols'][: -1], axis=1).values

    # define the regressor model
    model = xgb.XGBRegressor(
        min_child_weight=2,
        n_estimators=135,
        learning_rate=0.11,
        max_depth=6,
        booster='gbtree',
        verbosity=0,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='gpu_hist'
    )
    # model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, booster='gbtree', verbosity=0)
    # train the model
    model.fit(train_x, train_y)
    # generate predictions
    predictions = model.predict(test_x)

    # format data and write to file
    write_results_to_file(map(
        lambda indexed_pair: (indexed_pair[1][0], indexed_pair[1][1], predictions[indexed_pair[0]]),
        enumerate(test_set)
    ))


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run prediction
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
