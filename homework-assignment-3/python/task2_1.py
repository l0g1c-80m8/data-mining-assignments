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
    run_time_params['top_candidates'] = 15
    return run_time_params


def parse_dataset(filename):
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], (record[0], record[2]))) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], dict(business_set[1])))


def pearson_similarity(entry1, entry2):
    if entry1[0] == entry2[0]:
        return entry1[0], 0.0
    users1 = dict(entry1[1])
    users2 = dict(entry2[1])

    def _get_co_rated_avg(users):
        return reduce(
            lambda val, ele: float(val) + float(ele),
            map(
                lambda entry: entry[1],
                filter(lambda user: user[0] in co_rated_users, users.items())
            ),
            0
        ) / len(co_rated_users)

    co_rated_users = set(users1.keys()).intersection(users2.keys())
    users1_avg = _get_co_rated_avg(users1)
    users2_avg = _get_co_rated_avg(users2)

    numerator = reduce(
        lambda value, user_id: value + (float(users1[user_id]) - users1_avg) * (float(users2[user_id]) - users2_avg),
        co_rated_users,
        0.0
    )

    if numerator == 0:
        return entry1[0], 0.0

    denominator = sqrt(reduce(
        lambda value, user_id: value + pow((float(users1[user_id]) - users1_avg), 2),
        co_rated_users,
        0.0
    )) * sqrt(reduce(
        lambda value, user_id: value + pow((float(users2[user_id]) - users2_avg), 2),
        co_rated_users,
        0.0
    ))

    return entry1[0], numerator / denominator


def recommend(pair, dataset):
    business_id = pair[0]
    user_id = pair[1]

    if business_id not in dataset:
        return business_id, user_id, 0.0
    business_ratings = dataset[business_id]

    similar_businesses = sorted(map(
        lambda entry: pearson_similarity(entry, (business_id, business_ratings)),
        dataset.items()
    ),
        key=lambda business_similarity: business_similarity[1],
        reverse=True
    )[0:params['top_candidates']]

    prediction = reduce(
        lambda value, business_similarity: value + float(dataset[business_similarity[0]][user_id]) * business_similarity[1],
        similar_businesses,
        0.0
    ) / reduce(
        lambda value, business_similarity: value + business_similarity[1],
        similar_businesses,
        0.0
    )

    return business_id, user_id, prediction


def main():
    # dataset rdd
    dataset_rdd = parse_dataset(params['in_file'])
    dataset = dataset_rdd.collectAsMap()
    # test rdd
    test_rdd = parse_dataset(params['test_file']) \
        .flatMapValues(lambda val: val)

    test_rdd = test_rdd.map(lambda pair: recommend(pair, dataset))
    print(test_rdd.collect())


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run min-hashing + locality sensitive hashing
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
