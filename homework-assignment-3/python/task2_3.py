"""
Homework Assignment 3
Task 2 (Part 3)

Combine the recommendations systems developed in parts 1 and 2 into a hybrid system.
"""

import json
import os
import pandas as pd
import sys
import xgboost as xgb

from datetime import datetime
from functools import reduce
from math import sqrt
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'hw3-task2_1'
    run_time_params['in_dir'] = sys.argv[1]
    run_time_params['test_file'] = sys.argv[2]
    run_time_params['out_file'] = sys.argv[3]
    run_time_params['top_candidates'] = 20
    run_time_params['neighborhood_size'] = 20
    run_time_params['min_ratings'] = 200
    run_time_params['fallback_rating'] = 2.5
    run_time_params['train'] = 'yelp_train.csv'
    run_time_params['user'] = 'user.json'
    run_time_params['business'] = 'business.json'
    run_time_params['record_cols'] = ['user_id', 'business_id', 'rating']
    run_time_params['user_feature_cols'] = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']
    run_time_params['business_feature_cols'] = ['business_stars', 'business_review_count']
    run_time_params['alpha'] = 0.1
    return run_time_params


def dataset_average(mode):
    filename = '{}/{}'.format(params['in_dir'], params['train'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    key_idx = 0 if mode == 'users' else 1

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[key_idx], float(record[2]))) \
        .groupByKey() \
        .map(lambda key_set: (key_set[0], list(key_set[1]))) \
        .map(lambda key_set: (key_set[0], sum(key_set[1]) / len(key_set[1]))) \
        .collectAsMap()


def parse_dataset():
    filename = '{}/{}'.format(params['in_dir'], params['train'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], (record[0], float(record[2])))) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], dict(business_set[1])))


def parse_test_set():
    with open(params['test_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['test_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], record[0]))


def parse_train_set():
    filename = '{}/{}'.format(params['in_dir'], params['train'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_user_set():
    filename = '{}/{}'.format(params['in_dir'], params['user'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda user_obj: (user_obj[params['record_cols'][0]],
                               tuple(map(
                                   lambda col_name: user_obj[col_name],
                                   params['user_feature_cols']
                               ))
                               )
             )


def parse_business_set():
    filename = '{}/{}'.format(params['in_dir'], params['business'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda business_obj: (business_obj[params['record_cols'][1]],
                                   tuple(map(
                                       lambda col_name: business_obj[col_name.replace('business_', '')],
                                       params['business_feature_cols']
                                   ))
                                   )
             )


def pearson_similarity(entry1, entry2):
    if entry1[0] == entry2[0]:
        return None
    users1 = dict(entry1[1])
    users2 = dict(entry2[1])

    if entry2[2] not in users1:
        return None

    def _get_co_rated_avg(users):
        return reduce(
            lambda val, ele: val + ele,
            map(
                lambda entry: entry[1],
                filter(lambda user: user[0] in co_rated_users, users.items())
            ),
            0
        ) / len(co_rated_users)

    co_rated_users = set(users1.keys()).intersection(users2.keys())

    if len(co_rated_users) < params['top_candidates']:
        return entry1[0], 0.0

    users1_avg = _get_co_rated_avg(users1)
    users2_avg = _get_co_rated_avg(users2)

    numerator = reduce(
        lambda value, user_id: value + (users1[user_id] - users1_avg) * (users2[user_id] - users2_avg),
        co_rated_users,
        0.0
    )

    if numerator == 0:
        return entry1[0], 0.0

    denominator = sqrt(reduce(
        lambda value, user_id: value + pow((users1[user_id] - users1_avg), 2),
        co_rated_users,
        0.0
    )) * sqrt(reduce(
        lambda value, user_id: value + pow((users2[user_id] - users2_avg), 2),
        co_rated_users,
        0.0
    ))

    if denominator == 0:
        return entry1[0], 0.0

    return entry1[0], numerator / denominator


def recommend(pair, dataset, avg_user_ratings, avg_business_ratings):
    business_id = pair[0]
    user_id = pair[1]

    if business_id not in dataset:
        if business_id in avg_business_ratings:
            return business_id, user_id, avg_business_ratings[business_id]
        return business_id, user_id, params['fallback_rating']
    business_ratings = dataset[business_id]

    similar_businesses = sorted(filter(
        lambda entry: entry is not None and entry[1] > 0.0,
        map(
            lambda entry: pearson_similarity(entry, (business_id, business_ratings, user_id)),
            dataset.items()
        )
    ),
        key=lambda business_similarity: business_similarity[1],
        reverse=True
    )[0:params['neighborhood_size']]

    numerator = reduce(
        lambda value, business_similarity: value + dataset[business_similarity[0]][user_id] * business_similarity[1],
        similar_businesses,
        0.0
    )

    if numerator == 0.0:
        return business_id, user_id, avg_user_ratings[user_id]

    denominator = reduce(
        lambda value, business_similarity: value + business_similarity[1],
        similar_businesses,
        0.0
    )

    if denominator == 0.0:
        return business_id, user_id, avg_user_ratings[user_id]

    predicted_rating = numerator / denominator
    predicted_rating = max(0.0, predicted_rating)
    predicted_rating = min(5.0, predicted_rating)

    return business_id, user_id, predicted_rating


def write_results_to_file(predictions):
    file_header = 'user_id, business_id, prediction\n'
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for key in predictions:
            fh.write('{},{},{}\n'.format(key[0], key[1], predictions[key]))


def item_based_cf_prediction():
    # create the dataset with (not cold) business entries and their ratings
    dataset = parse_dataset() \
        .filter(lambda business_set: len(business_set[1]) >= params['min_ratings']) \
        .collectAsMap()

    # get averages for users and business to be used as fallbacks for cold businesses and missing/new businesses/users
    avg_user_ratings, avg_business_ratings = dataset_average('users'), dataset_average('businesses')

    # results rdd
    return sc.parallelize(parse_test_set().collect()) \
        .map(lambda pair: recommend(pair, dataset, avg_user_ratings, avg_business_ratings)) \
        .map(lambda triple: ((triple[1], triple[0]), triple[2])) \
        .collectAsMap()


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], tuple([0] * len(params['user_feature_cols'])))
    business_features = business_data.get(record[1], tuple([0] * len(params['business_feature_cols'])))
    return record + user_features + business_features


def model_based_prediction():
    # create feature data
    user_data = parse_user_set().collectAsMap()
    business_data = parse_business_set().collectAsMap()

    # create training dataset
    train_df = pd.DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=params['record_cols'] + params['user_feature_cols'] + params['business_feature_cols']
    )

    # create test dataset
    test_set = parse_test_set() \
        .map(lambda record: (record[1], record[0])) \
        .map(lambda record: fill_features(record, user_data, business_data)) \
        .collect()
    test_df = pd.DataFrame(test_set,
                           columns=params['record_cols'][: -1] + params['user_feature_cols'] + params[
                               'business_feature_cols']
                           )

    # define the regressor model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, booster='gbtree', verbosity=0)
    # train the model
    model.fit(train_df.drop(params['record_cols'], axis=1).values, train_df[params['record_cols'][-1:]].values)
    # generate predictions
    predictions = model.predict(test_df.drop(params['record_cols'][: -1], axis=1).values)

    return dict(map(
        lambda indexed_pair: ((indexed_pair[1][0], indexed_pair[1][1]), predictions[indexed_pair[0]]),
        enumerate(test_set)
    ))


def main():
    # Item-based CF rating prediction
    item_based_predictions = item_based_cf_prediction()

    # model (regressor) based rating prediction
    model_based_predictions = model_based_prediction()

    # combine ratings
    combined_predictions = dict()
    for key in item_based_predictions.keys():
        combined_predictions[key] = \
            params['alpha'] * item_based_predictions.get(key, params['fallback_rating']) \
            + (1 - params['alpha']) * model_based_predictions.get(key, params['fallback_rating'])

    # write results to the output file
    write_results_to_file(combined_predictions)


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
