"""
Homework Assignment 3
Task 1

Locality Sensitive Hashing on signature matrix computed by min-hashing user / business review data to identify similar businesses.

Representation: Businesses (sets) are represented in the dimension of users (elements).
"""

import os
import sys

from itertools import chain
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'hw3-task1'
    run_time_params['in_file'] = sys.argv[1]
    run_time_params['out_file'] = sys.argv[2]
    return run_time_params


def parse_dataset():
    with open(params['in_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['in_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], record[0])) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], set(business_set[1])))


def get_mappings():
    business_ids = dataset_rdd.map(lambda business_set: business_set[0]).collect()
    user_ids = list(chain.from_iterable(dataset_rdd.map(lambda business_set: business_set[1]).collect()))

    def map_by_counter(ids, _map, counter):
        for _id in ids:
            if _id not in _map:
                _map[_id] = counter
                counter += 1
        return _map

    return map_by_counter(business_ids, dict(), 0), map_by_counter(user_ids, dict(), 0)


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # dataset rdd
    dataset_rdd = parse_dataset()

    # get element (row / users) and set (cols / businesses) mappings
    element_map, set_map = get_mappings()
    print(len(element_map), len(set_map))
