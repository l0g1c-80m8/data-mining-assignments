"""
Homework Assignment 3
Task 2 (Part 1)

An item-based collaborative filtering mechanism to generate recommendation of businesses (items) for users.
"""

import os
import sys

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
    run_time_params['in_file'] = sys.argv[1]
    run_time_params['test_file'] = sys.argv[2]
    run_time_params['out_file'] = sys.argv[3]
    run_time_params['top_candidates'] = 20
    run_time_params['neighborhood_size'] = 20
    run_time_params['min_ratings'] = 100
    run_time_params['fallback_rating'] = 2.5
    return run_time_params


def dataset_average(mode):
    with open(params['in_file'], 'r') as fh:
        header = fh.readline().strip()

    key_idx = 0 if mode == 'users' else 1

    return sc.textFile(params['in_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[key_idx], float(record[2]))) \
        .groupByKey() \
        .map(lambda key_set: (key_set[0], list(key_set[1]))) \
        .map(lambda key_set: (key_set[0], sum(key_set[1]) / len(key_set[1]))) \
        .collectAsMap()


def parse_dataset():
    with open(params['in_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['in_file']) \
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


def write_results_to_file(recommendations):
    file_header = 'user_id, business_id, prediction\n'
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for triple in recommendations:
            fh.write('{},{},{}\n'.format(triple[1], triple[0], triple[2]))


def main():
    # create the dataset with (not cold) business entries and their ratings
    dataset = parse_dataset() \
        .filter(lambda business_set: len(business_set[1]) >= params['min_ratings']) \
        .collectAsMap()

    # get averages for users and business to be used as fallbacks for cold businesses and missing/new businesses/users
    avg_user_ratings, avg_business_ratings = dataset_average('users'), dataset_average('businesses')

    # results rdd
    results_rdd = sc.parallelize(parse_test_set().collect()) \
        .map(lambda pair: recommend(pair, dataset, avg_user_ratings, avg_business_ratings))

    # write output to file
    write_results_to_file(results_rdd.collect())


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
