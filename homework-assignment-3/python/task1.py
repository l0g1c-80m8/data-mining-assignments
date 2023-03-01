"""
Homework Assignment 3
Task 1

Locality Sensitive Hashing on signature matrix computed by min-hashing user / business review data to identify similar businesses.

Representation: Businesses (sets) are represented in the dimension of users (elements).
"""

import os
import sys

from collections import namedtuple
from datetime import datetime
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
    run_time_params['bands'] = 1
    run_time_params['n_hashers'] = 2
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


def get_mappings(dataset_rdd):
    business_ids = dataset_rdd.map(lambda business_set: business_set[0]).collect()
    user_ids = list(chain.from_iterable(dataset_rdd.map(lambda business_set: business_set[1]).collect()))

    def map_by_counter(ids, _map, counter):
        for _id in ids:
            if _id not in _map:
                _map[_id] = counter
                counter += 1
        return _map

    return map_by_counter(business_ids, dict(), 0), map_by_counter(user_ids, dict(), 0)


def construct_hashers(num_rows):
    HasherParams = namedtuple('HasherParams', ['a', 'b', 'm'])
    hasher_params_list = [
        HasherParams(a=1, b=1, m=num_rows),
        HasherParams(a=3, b=1, m=num_rows)
    ]

    # https://stackoverflow.com/a/25104050/16112875
    return [
        lambda x: (hash_params.a * x + hash_params.b) % hash_params.m
        for hash_params in hasher_params_list
    ]


def get_signature(dataset_rdd, hashers, element_map):
    def _find_min_sig(id_set_pair):
        _id = id_set_pair[0]
        _set = id_set_pair[1]

        sig_vector = list()
        for hasher in hashers:
            sig_val = sys.maxsize
            for element in _set:
                ele_val = hasher(element_map[element])
                sig_val = min(sig_val, ele_val)
            sig_vector.append(sig_val)
        return _id, sig_vector

    return dataset_rdd.map(_find_min_sig)


def main():
    # dataset rdd
    dataset_rdd = parse_dataset()

    # get element (row / users) and set (cols / businesses) mappings
    set_map, element_map = get_mappings(dataset_rdd)

    # define a collection of hash functions
    hashers = construct_hashers(len(element_map))

    # construct a column-wise signature matrix
    signature = get_signature(dataset_rdd, hashers, element_map)
    print(signature.collect())


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
